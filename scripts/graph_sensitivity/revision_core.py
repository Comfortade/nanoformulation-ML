# %% imports
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import platform
import random
import time
import warnings
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from openpyxl import load_workbook
from scipy.optimize import least_squares
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


def show(frame):
    try:
        from IPython.display import display
    except ImportError:
        print(frame.to_string(index=False) if isinstance(frame, pd.DataFrame) else frame)
    else:
        display(frame)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str), encoding='utf-8')

# %% workbook
def read_workbook(path):
    """Read the latest layout only, with exact cells recorded for every record."""
    formulas = load_workbook(path, data_only=False)
    values = load_workbook(path, data_only=True)
    required = {'Conc.', 'Releases', 'EE&LC', 'Mean', 'Release pH 5.5',
                'Release pH 6.5', 'Relaese pH 7.4'}
    if not required.issubset(formulas.sheetnames):
        raise ValueError('This is not the latest workbook layout. Missing tabs: '
                         + ', '.join(sorted(required - set(formulas.sheetnames)))
                         + '. Use the updated Excel supplied for the revision.')
    concentration_rows = []
    # Validated against the supplied workbook. Do not combine other tabs as new readings.
    blocks = [(5.5, 'Test', 3), (5.5, 'Control', 8), (6.5, 'Test', 13),
              (6.5, 'Control', 18), (7.4, 'Test', 23), (7.4, 'Control', 28)]
    for ph, form, first in blocks:
        sheet = formulas['Conc.']
        if sheet.cell(4, first).value != form:
            raise ValueError(f'Concentration header changed at column {first}')
        source = formulas['pH' + str(ph)]
        od_start = 3 if form == 'Test' else 10
        for row in range(6, 12):
            t = float(values['Conc.'].cell(row, 2).value)
            for rep in range(3):
                cell = values['Conc.'].cell(row, first + rep)
                y = float(cell.value)
                od_cell = source.cell(row, od_start + rep)
                calculated = (float(od_cell.value) - 0.0017) / 0.0182
                if not np.isclose(y, calculated, rtol=1e-10, atol=1e-12):
                    raise ValueError(f'Concentration disagrees with source OD at {cell.coordinate}')
                concentration_rows.append(dict(Time_h=t, pH=ph, Formulation=form,
                    Replicate=rep + 1, value=y, source_sheet='Conc.', source_cell=cell.coordinate,
                    OD=float(od_cell.value), OD_sheet=source.title, OD_cell=od_cell.coordinate))
    concentration = pd.DataFrame(concentration_rows)
    if len(concentration) != 108 or concentration.duplicated(
            ['Time_h', 'pH', 'Formulation', 'Replicate']).any():
        raise ValueError('Unexpected concentration record count or repeated identifiers')
    summary = concentration.groupby(['Time_h', 'pH', 'Formulation'], as_index=False).agg(
        mean=('value', 'mean'), SD=('value', 'std'), n=('value', 'size'))
    # Check all concentration mean ± sample-SD strings in the supplied release tabs.
    summary_checks = []
    for ph, form, _ in blocks:
        name, first = {5.5: ('Release pH 5.5', 9), 6.5: ('Release pH 6.5', 6),
                       7.4: ('Relaese pH 7.4', 6)}[ph]
        column = 4 if form == 'Test' else 9
        subset = summary[(summary.pH == ph) & (summary.Formulation == form)].sort_values('Time_h')
        for i, record in enumerate(subset.to_dict('records')):
            c = values[name].cell(first + i, column)
            tokens = str(c.value).split('±')
            if len(tokens) != 2:
                raise ValueError(f'Cannot interpret mean/SD at {name}!{c.coordinate}')
            m, sd = [float(x.strip()) for x in tokens]
            ok = abs(m - record['mean']) <= 5.1e-6 and abs(sd - record['SD']) <= 5.1e-6
            summary_checks.append(record | {'source': name + '!' + c.coordinate, 'matches_rounding': ok})
            if not ok:
                raise ValueError(f'Mean or sample SD mismatch at {name}!{c.coordinate}')
    release_rows = []
    release_checks = []
    for ph, form, col in [(5.5, 'Test', 3), (5.5, 'Control', 6), (6.5, 'Test', 9),
                          (6.5, 'Control', 12), (7.4, 'Test', 15), (7.4, 'Control', 18)]:
        for row in range(9, 15):
            sheet = values['Releases']
            t = float(sheet.cell(row, 2).value)
            mean = float(sheet.cell(row, col).value)
            mass = float(sheet.cell(row, col + 1).value)
            fcell = formulas['Releases'].cell(row, col + 2)
            formula = str(fcell.value)
            # Formula denominator is read from Excel, never chosen or replaced by this notebook.
            if not formula.startswith('=') or '/' not in formula:
                raise ValueError(f'Expected release formula at {fcell.coordinate}')
            denominator = float(formula.rsplit('/', 1)[1].strip())
            expected = mass * 100.0 / denominator
            cached = sheet.cell(row, col + 2).value
            if cached is None or not np.isclose(float(cached), expected, rtol=1e-10, atol=1e-12):
                raise ValueError('Missing/stale release formula value. Open and save the workbook in Excel: '
                                 + fcell.coordinate)
            source_mean = summary[(summary.Time_h == t) & (summary.pH == ph)
                                  & (summary.Formulation == form)]['mean'].item()
            if not np.isclose(mean, source_mean, rtol=1e-10, atol=1e-12):
                raise ValueError('Release mean differs from triplicates')
            release_rows.append(dict(Time_h=t, pH=ph, Formulation=form, value=float(cached),
                source_sheet='Releases', source_cell=fcell.coordinate,
                concentration_mean=mean, workbook_mass_value=mass,
                workbook_denominator=denominator, workbook_formula=formula))
            release_checks.append(dict(Time_h=t, pH=ph, Formulation=form,
                formula=formula, formula_result=expected, cached_result=float(cached), matches=True))
    release = pd.DataFrame(release_rows)
    if len(release) != 36 or release.duplicated(['Time_h', 'pH', 'Formulation']).any():
        raise ValueError('Expected exactly 36 cumulative-release summaries')
    for frame in (concentration, release):
        if not np.isfinite(frame[['Time_h', 'pH', 'value']].to_numpy()).all():
            raise ValueError('Nonfinite experimental data')
        if set(frame.Time_h) != {0, 6, 12, 24, 48, 96}:
            raise ValueError('Sampling times changed')
        frame['Formulation_binary'] = frame.Formulation.map({'Control': 0, 'Test': 1})
        frame['cell_group'] = frame.groupby(['Time_h', 'pH', 'Formulation']).ngroup()
        frame['profile_group'] = frame.groupby(['pH', 'Formulation']).ngroup()
        frame['row_id'] = np.arange(len(frame))
    load_rows = list(values['EE&LC'].values)
    loading = pd.DataFrame([list(r)[:5] for r in load_rows[2:] if r[0] is not None],
        columns=['record', 'pH', 'citral_retained_as_supplied', 'EE_percent_as_supplied', 'LC_percent_as_supplied'])
    return concentration, summary, release, loading, pd.DataFrame(summary_checks), pd.DataFrame(release_checks)

# %% models
FEATURES = ['Time_h', 'pH', 'Formulation_binary']
CITRAL_SMILES = 'CC(=CCCC(=CC=O)C)C'


def citral_graph():
    """Citral connectivity only; stereochemistry and mixture composition are not inferred."""
    mol = Chem.MolFromSmiles(CITRAL_SMILES)
    if mol is None or rdMolDescriptors.CalcMolFormula(mol) != 'C10H16O':
        raise ValueError('Citral graph failed structural sanity check')
    x = np.array([[a.GetAtomicNum() / 10., a.GetDegree() / 4.,
                   a.GetTotalNumHs() / 4., float(a.GetIsAromatic())]
                  for a in mol.GetAtoms()], dtype=np.float32)
    a = np.eye(mol.GetNumAtoms(), dtype=np.float32)
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        a[i, j] = a[j, i] = float(b.GetBondTypeAsDouble())
    d = 1. / np.sqrt(a.sum(axis=1))
    a = d[:, None] * a * d[None, :]
    return torch.tensor(x), torch.tensor(a), dict(smiles=CITRAL_SMILES,
        formula=rdMolDescriptors.CalcMolFormula(mol), atom_count=mol.GetNumAtoms(),
        bond_count=mol.GetNumBonds(), molecular_weight=Descriptors.MolWt(mol),
        source='Repository citral connectivity; formula cross-check against PubChem CID 638011',
        limitation='One fixed citral graph in both formulations; no chemical-diversity or compound-importance inference.')


class NeuralSurrogate(nn.Module):
    def __init__(self, graph=False, kinetic=False, target_scale=1.0):
        super().__init__()
        self.graph = graph
        self.kinetic = kinetic
        self.register_buffer('capacity_scale', torch.tensor(100. / target_scale))
        if graph:
            x, a, _ = citral_graph()
            self.register_buffer('atoms', x)
            self.register_buffer('adjacency', a)
            self.g1 = nn.Linear(x.shape[1], 8)
            self.g2 = nn.Linear(8, 8)
        environment_width = 2 if kinetic else 3
        self.head = nn.Sequential(nn.Linear(environment_width + (8 if graph else 0), 16), nn.Tanh(),
                                  nn.Linear(16, 8), nn.Tanh(), nn.Linear(8, 2 if kinetic else 1))
        if kinetic:
            # Start near the scale of the training response, rather than 50 percentage points.
            initial_fraction = np.clip(0.5 * target_scale / 100., 1e-8, 1 - 1e-8)
            with torch.no_grad():
                self.head[-1].bias[0] = float(np.log(initial_fraction / (1 - initial_fraction)))
                self.head[-1].bias[1] = 20.

    def forward(self, x, time_normalized=None):
        if self.kinetic:
            # A and k depend only on pH, formulation and fixed chemistry, not time.
            x = x[:, 1:]
        if self.graph:
            h = torch.tanh(self.g1(self.adjacency @ self.atoms))
            h = torch.tanh(self.g2(self.adjacency @ h)).mean(dim=0)
            x = torch.cat([x, h.unsqueeze(0).expand(len(x), -1)], dim=1)
        out = self.head(x)
        if self.kinetic:
            if time_normalized is None:
                raise ValueError('Kinetically constrained model requires unscaled nonnegative time')
            amplitude = self.capacity_scale * torch.sigmoid(out[:, 0])
            rate = torch.nn.functional.softplus(out[:, 1]) + 1e-6
            return amplitude * (-torch.expm1(-rate * time_normalized))
        return out.flatten()


def fit_curve(t, y, kind='saturating'):
    """Fit only supplied training records; average technical rows within time first."""
    tmp = pd.DataFrame({'t': t, 'y': y}).groupby('t', as_index=False).y.mean()
    t, y = tmp.t.to_numpy() / 96., tmp.y.to_numpy()
    scale = max(float(np.max(np.abs(y))), 1e-8)
    if kind == 'higuchi':
        x = np.sqrt(t)
        coefficient = max(float(x @ y / max(x @ x, 1e-12)), 0.)
        return lambda times: coefficient * np.sqrt(np.maximum(np.asarray(times), 0.) / 96.)
    if len(np.unique(t[t > 0])) < 2:
        # Two saturating parameters cannot be identified from one positive time.
        coefficient = max(float(t @ y / max(t @ t, 1e-12)), 0.)
        return lambda times: coefficient * np.maximum(np.asarray(times), 0.) / 96.
    def residual(theta):
        return (theta[0] * (-np.expm1(-theta[1] * t)) - y) / scale
    fits = [least_squares(residual, x0=[scale, k], bounds=([0., 1e-6], [100., 1000.]),
                          max_nfev=2000) for k in [1., 10., 100.]]
    successful = [f for f in fits if f.success and np.isfinite(f.cost)]
    if not successful:
        raise RuntimeError('Saturating curve did not converge')
    best = min(successful, key=lambda f: f.cost)
    amplitude, rate = best.x
    return lambda times: amplitude * (-np.expm1(-rate * np.maximum(np.asarray(times), 0.) / 96.))


def curve_predictions(train, query, kind='saturating'):
    cache = {}
    out, fallbacks = [], []
    for record in query.to_dict('records'):
        same = train[(train.pH == record['pH']) & (train.Formulation == record['Formulation'])]
        route = 'same_profile'
        if same.empty:
            same = train[train.pH == record['pH']]
            route = 'same_pH_other_formulation'
        if same.empty:
            same = train
            route = 'pooled_training_profiles'
        key = (route, float(record['pH']), record['Formulation'])
        # Cache is local to this call and contains training-only fitted curves.
        if key not in cache:
            cache[key] = fit_curve(same.Time_h.to_numpy(), same.value.to_numpy(), kind)
        out.append(float(cache[key]([record['Time_h']])[0]))
        fallbacks.append(route)
    return np.array(out), fallbacks


def fit_neural(train, model_name, seed, config, target):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    scaler = StandardScaler().fit(train[FEATURES])
    x = torch.tensor(scaler.transform(train[FEATURES]), dtype=torch.float32)
    yscale = max(float(np.max(np.abs(train.value))), 1e-8)
    y = torch.tensor(train.value.to_numpy() / yscale, dtype=torch.float32)
    is_graph = model_name.startswith('GNN')
    regularized = model_name.endswith('_kinetic')
    model = NeuralSurrogate(graph=is_graph, kinetic=regularized, target_scale=yscale)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'])
    prior = None
    if regularized:
        if target != 'cumulative_percent_as_supplied':
            raise ValueError('No cumulative-release kinetic prior is imposed on concentration')
        prior_values, _ = curve_predictions(train, train)
        prior = torch.tensor(prior_values / yscale, dtype=torch.float32)
    model.train()
    for epoch in range(config['epochs']):
        optimizer.zero_grad()
        predicted = model(x, torch.tensor(train.Time_h.to_numpy() / 96., dtype=torch.float32))
        loss = ((predicted - y) ** 2).mean()
        if regularized:
            loss = loss + config['kinetic_weight'] * ((predicted - prior) ** 2).mean()
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite neural training loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
    model.eval()
    def predict(query):
        with torch.no_grad():
            return model(torch.tensor(scaler.transform(query[FEATURES]), dtype=torch.float32),
                         torch.tensor(query.Time_h.to_numpy() / 96., dtype=torch.float32)).numpy() * yscale
    return predict, model, scaler, yscale, float(loss.detach())

# %% evaluation
def splits_for(frame, target):
    if target == 'concentration_as_supplied':
        # Replicates of each time/pH/formulation cell cannot cross the split.
        for fold, (tr, te) in enumerate(GroupKFold(n_splits=5).split(frame, groups=frame.cell_group)):
            yield 'grouped_conditions', fold, tr, te
    else:
        # Cumulative values within a profile share concentration history: keep a full profile together.
        for fold, (tr, te) in enumerate(LeaveOneGroupOut().split(frame, groups=frame.profile_group)):
            yield 'held_out_profiles', fold, tr, te
    # Forecast from past observations only; never train on a later cumulative target.
    for fold, t in enumerate([12., 24., 48., 96.]):
        yield 'forward_time', fold, np.flatnonzero(frame.Time_h.to_numpy() < t), np.flatnonzero(frame.Time_h.to_numpy() == t)


def metrics_for(part):
    actual, predicted = part.observed.to_numpy(), part.predicted.to_numpy()
    return dict(n=len(part), R2=float(r2_score(actual, predicted)) if len(part) > 1 and np.var(actual) > 1e-24 else np.nan,
        RMSE=float(np.sqrt(mean_squared_error(actual, predicted))),
        MAE=float(mean_absolute_error(actual, predicted)))


def audit_splits(datasets, output):
    records = []
    assert 'value' not in FEATURES and 'Replicate' not in FEATURES
    for target, frame in datasets.items():
        for protocol, fold, tr, te in splits_for(frame, target):
            train, test = frame.iloc[tr], frame.iloc[te]
            row_overlap = len(set(train.row_id) & set(test.row_id))
            cell_overlap = len(set(train.cell_group) & set(test.cell_group))
            profile_overlap = len(set(train.profile_group) & set(test.profile_group))
            future_in_training = bool(train.Time_h.max() >= test.Time_h.min()) if protocol == 'forward_time' else False
            assert row_overlap == 0 and cell_overlap == 0
            if protocol == 'held_out_profiles':
                assert profile_overlap == 0
            assert not future_in_training
            # Verify the scaler's moments come from the training features, not a combined table.
            scaler = StandardScaler().fit(train[FEATURES])
            assert np.allclose(scaler.mean_, train[FEATURES].mean().to_numpy())
            records.append(dict(target=target, protocol=protocol, fold=fold,
                train_rows=len(tr), test_rows=len(te), overlapping_rows=row_overlap,
                overlapping_condition_cells=cell_overlap, overlapping_profiles=profile_overlap,
                future_time_in_training=future_in_training, train_only_scaler_check=True,
                note=('Shared profiles are allowed for concentration/forward-time tasks; '
                      'complete profiles are separated for cumulative held-out-profile validation.')))
    table = pd.DataFrame(records)
    table.to_csv(output / 'split_leakage_checks.csv', index=False)
    return table


def evaluate(datasets, config, output):
    rows, assignments, warnings_log, timing, fallback_log = [], [], [], [], []
    for target, frame in datasets.items():
        models = ['linear', 'random_forest', 'MLP', 'GNN']
        if target == 'cumulative_percent_as_supplied':
            models += ['saturating_exponential', 'higuchi', 'MLP_kinetic', 'GNN_kinetic']
        for protocol, fold, tr, te in splits_for(frame, target):
            train, test = frame.iloc[tr].copy(), frame.iloc[te].copy()
            # Prediction receives covariates only, never held-out outcomes or source measurements.
            query = test[FEATURES + ['Formulation']].copy()
            assert 'value' not in query and 'concentration_mean' not in query
            assert set(train.row_id).isdisjoint(test.row_id)
            if protocol == 'held_out_profiles':
                assert set(train.profile_group).isdisjoint(test.profile_group)
            else:
                assert set(train.cell_group).isdisjoint(test.cell_group)
            if protocol == 'forward_time':
                assert train.Time_h.max() < test.Time_h.min()
            for role, indices in [('train', tr), ('test', te)]:
                for idx in indices:
                    assignments.append(dict(target=target, protocol=protocol, fold=fold, role=role,
                        row_id=int(frame.iloc[idx].row_id), cell_group=int(frame.iloc[idx].cell_group),
                        profile_group=int(frame.iloc[idx].profile_group), Time_h=float(frame.iloc[idx].Time_h)))
            for name in models:
                seeds = config['seeds'] if name in ['random_forest', 'MLP', 'GNN', 'MLP_kinetic', 'GNN_kinetic'] else [config['seeds'][0]]
                for seed in seeds:
                    start = time.perf_counter()
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always')
                        if name == 'linear':
                            model = make_pipeline(StandardScaler(), LinearRegression())
                            model.fit(train[FEATURES], train.value)
                            pred = model.predict(query[FEATURES])
                        elif name == 'random_forest':
                            model = RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                                          random_state=seed, n_jobs=1)
                            model.fit(train[FEATURES], train.value)
                            pred = model.predict(query[FEATURES])
                        elif name in ['saturating_exponential', 'higuchi']:
                            kind = 'higuchi' if name == 'higuchi' else 'saturating'
                            pred, routes = curve_predictions(train, query, kind)
                            for record, route in zip(test.to_dict('records'), routes):
                                fallback_log.append(dict(target=target, protocol=protocol, fold=fold,
                                    model=name, row_id=record['row_id'], route=route))
                        else:
                            predict, _, _, _, loss = fit_neural(train, name, seed, config, target)
                            pred = predict(query)
                    for w in caught:
                        warnings_log.append(dict(target=target, protocol=protocol, fold=fold,
                            model=name, seed=seed, category=w.category.__name__, message=str(w.message)))
                    if len(pred) != len(test) or not np.isfinite(pred).all():
                        raise RuntimeError('Missing or nonfinite held-out predictions')
                    timing.append(dict(target=target, protocol=protocol, fold=fold, model=name,
                                       seed=seed, seconds=time.perf_counter() - start))
                    for record, value in zip(test.to_dict('records'), pred):
                        rows.append(dict(target=target, protocol=protocol, fold=fold, model=name,
                            seed=seed, row_id=record['row_id'], Time_h=record['Time_h'], pH=record['pH'],
                            Formulation=record['Formulation'], observed=record['value'], predicted=float(value)))
            print(f'{target} | {protocol} | fold {fold + 1}: completed', flush=True)
    results = pd.DataFrame(rows)
    if results.duplicated(['target', 'protocol', 'model', 'seed', 'row_id']).any():
        raise AssertionError('Repeated held-out predictions')
    results['residual'] = results.observed - results.predicted
    for key, part in results.groupby(['target', 'protocol', 'model', 'seed']):
        frame = datasets[key[0]]
        expected = len(frame) if key[1] != 'forward_time' else int(frame.Time_h.isin([12, 24, 48, 96]).sum())
        assert len(part) == expected
    results.to_csv(output / 'held_out_predictions.csv', index=False)
    pd.DataFrame(assignments).to_csv(output / 'split_assignments.csv', index=False)
    pd.DataFrame(timing).to_csv(output / 'runtime.csv', index=False)
    pd.DataFrame(fallback_log).to_csv(output / 'kinetic_fit_fallbacks.csv', index=False)
    write_json(output / 'training_warnings.json', warnings_log)
    records = []
    keys = ['target', 'protocol', 'model', 'seed']
    for extra, scope in [([], 'overall'), (['pH'], 'pH'), (['Formulation'], 'formulation'),
                         (['pH', 'Formulation'], 'pH_formulation'), (['fold'], 'fold')]:
        columns = keys + extra
        for key, part in results.groupby(columns):
            records.append(dict(zip(columns, key)) | dict(scope=scope) | metrics_for(part))
    metrics = pd.DataFrame(records)
    metrics.to_csv(output / 'metrics_by_seed.csv', index=False)
    overall = metrics[metrics.scope == 'overall'].groupby(['target', 'protocol', 'model']).agg(
        seed_count=('seed', 'size'), R2_mean=('R2', 'mean'), R2_seed_SD=('R2', 'std'),
        RMSE_mean=('RMSE', 'mean'), MAE_mean=('MAE', 'mean')).reset_index()
    overall.to_csv(output / 'model_comparison.csv', index=False)
    return results, metrics, overall

# %% reporting
def diagnostics(results, release, output):
    plot_files = []
    for (target, protocol), part in results.groupby(['target', 'protocol']):
        models = sorted(part.model.unique())
        fig, axes = plt.subplots(2, len(models), figsize=(max(12, 3 * len(models)), 6), squeeze=False)
        for j, name in enumerate(models):
            points = part[part.model == name].groupby('row_id')[['observed', 'predicted']].mean()
            axes[0, j].scatter(points.observed, points.predicted, s=12)
            lo, hi = points.observed.min(), points.observed.max()
            axes[0, j].plot([lo, hi], [lo, hi], '--', color='black')
            axes[0, j].set(title=name, xlabel='Observed', ylabel='Held-out prediction')
            axes[1, j].scatter(points.predicted, points.observed - points.predicted, s=12)
            axes[1, j].axhline(0, linestyle='--', color='black')
            axes[1, j].set(xlabel='Held-out prediction', ylabel='Residual')
        fig.suptitle(target + ' / ' + protocol + '\nPlot points average available seeds; metrics are computed per seed.')
        fig.tight_layout()
        path = output / f'{target}_{protocol}.png'
        fig.savefig(path, dpi=140)
        plt.close(fig)
        plot_files.append(path)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (ph, part) in zip(axes, release.groupby('pH')):
        for form, g in part.groupby('Formulation'):
            g = g.sort_values('Time_h')
            ax.plot(g.Time_h, g.value, 'o-', label=form)
        ax.set(title=f'pH {ph}', xlabel='Time (h)', ylabel='Cumulative release (% as supplied)')
        ax.legend()
    fig.tight_layout()
    path = output / 'excel_release_profiles.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    plot_files.append(path)
    return plot_files


def final_interpolation(release, config, output):
    # A full-data fit is used only for exploratory curves, never for a validation score.
    records, prediction_flags = [], []
    checkpoint_dir = output / 'final_models'
    checkpoint_dir.mkdir(exist_ok=True)
    for seed in config['seeds']:
        predict, model, scaler, yscale, loss = fit_neural(release, 'GNN_kinetic', seed, config,
                                                        'cumulative_percent_as_supplied')
        torch.save({'state_dict': model.state_dict(), 'scaler_mean': scaler.mean_.tolist(),
                    'scaler_scale': scaler.scale_.tolist(), 'target_scale': yscale,
                    'features': FEATURES, 'config': config, 'citral_smiles': CITRAL_SMILES},
                   checkpoint_dir / f'gnn_kinetic_seed_{seed}.pt')
        for ph in [5.5, 6.3, 6.5, 7.4]:
            for form, binary in [('Control', 0), ('Test', 1)]:
                query = pd.DataFrame({'Time_h': np.linspace(0, 120, 121), 'pH': ph,
                                      'Formulation_binary': binary})
                pred = predict(query)
                for t, value in zip(query.Time_h, pred):
                    records.append(dict(seed=seed, pH=ph, Formulation=form, Time_h=t,
                        predicted_percent_as_supplied=float(value),
                        status=('unvalidated_pH_interpolation' if ph == 6.3 else 'full_data_fit')
                               + (';time_extrapolation' if t > 96 else '')))
                prediction_flags.append(dict(seed=seed, pH=ph, Formulation=form,
                    negative_prediction=bool((pred < 0).any()), over_100_prediction=bool((pred > 100).any()),
                    decreasing_segments=int((np.diff(pred) < -1e-8).sum())))
    curves = pd.DataFrame(records)
    curves.to_csv(output / 'exploratory_curves_not_validation.csv', index=False)
    pd.DataFrame(prediction_flags).to_csv(output / 'exploratory_physical_checks.csv', index=False)
    return curves


def run_analysis(excel_path, output, config):
    start = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    concentration, summary, release, loading, mean_checks, formula_checks = read_workbook(excel_path)
    concentration.to_csv(output / 'concentration_readings.csv', index=False)
    summary.to_csv(output / 'concentration_mean_SD.csv', index=False)
    release.to_csv(output / 'release_summaries_as_supplied.csv', index=False)
    loading.to_csv(output / 'loading_as_supplied.csv', index=False)
    mean_checks.to_csv(output / 'mean_SD_checks.csv', index=False)
    formula_checks.to_csv(output / 'excel_formula_checks.csv', index=False)
    _, _, graph_metadata = citral_graph()
    versions = {name: importlib.metadata.version(name) for name in
                ['numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib', 'openpyxl', 'torch', 'rdkit']}
    config = dict(config)
    write_json(output / 'run_config.json', config)
    write_json(output / 'model_specifications.json', dict(
        features=FEATURES,
        linear='StandardScaler fitted inside fold, then ordinary least squares with intercept',
        random_forest=dict(n_estimators=200, min_samples_leaf=2, n_jobs=1,
                           other_parameters='scikit-learn version-specific defaults'),
        neural=dict(head_hidden=[16, 8], activation='tanh',
                    response_scaling='divide by maximum absolute TRAINING target',
                    feature_scaling='StandardScaler fitted on TRAINING covariates',
                    dropout=0, early_stopping=False, optimizer='Adam',
                    gradient_clip_norm=5.0, epochs=config['epochs'],
                    learning_rate=config['learning_rate'], weight_decay=config['weight_decay']),
        graph=dict(layers=2, hidden_features=8, pooling='mean',
                   convolution='Linear(D^-1/2 (A+I) D^-1/2 H), then tanh; bond-order-weighted adjacency',
                   node_features=['atomic_number/10', 'degree/4', 'hydrogen_count/4', 'aromatic_indicator']),
        kinetic=dict(shape='A*(1-exp(-k*time_h/96)); A=100*sigmoid(head_0), k=softplus(head_1)+1e-6',
                     parameter_inputs=['pH', 'formulation', 'fixed graph when enabled'],
                     loss='mean squared scaled error + kinetic_weight*mean squared scaled training-prior discrepancy',
                     kinetic_weight=config['kinetic_weight'], prior='training-only fitted saturating exponential'),
        classical='Training-only nonnegative fits on time means; same-profile/same-pH/pooled fallback routes saved'))
    (output / 'requirements_recorded.txt').write_text(
        '# Actual versions used; CPU torch wheels may require https://download.pytorch.org/whl/cpu\n'
        + '\n'.join(f'{name}=={version}' for name, version in versions.items()) + '\n', encoding='utf-8')
    write_json(output / 'data_manifest.json', dict(source_file=excel_path.name,
        sha256=hashlib.sha256(excel_path.read_bytes()).hexdigest(),
        sheet_names=load_workbook(excel_path, read_only=True).sheetnames,
        concentration_rows=len(concentration), cumulative_summary_rows=len(release),
        concentration_units='Supplied measurement scale; no unit conversion is performed',
        release_units='Percentage values exactly as calculated in Excel; no extra multiplication by 100',
        source_policy='Excel is the analysis source. The original workbook is never written to.',
        versions=versions, python=platform.python_version(), graph=graph_metadata))
    show(summary.head(12))
    print('108 concentration entries; 36 cumulative-release summaries. No duplicated summaries.')
    datasets = {'concentration_as_supplied': concentration,
                'cumulative_percent_as_supplied': release}
    audit_splits(datasets, output)
    results, metrics, comparison = evaluate(datasets, config, output)
    plot_files = diagnostics(results, release, output)
    curves = final_interpolation(release, config, output)
    report = [
        '# Analysis notes for manuscript revision',
        '',
        'Results are generated from the selected updated Excel; its source hash, cells and formulas are saved.',
        'Concentrations and cumulative-release summaries are distinct targets and must not be pooled.',
        'No source measurements, denominators, units or release percentages are silently changed.',
        'Baseline/GNN comparisons use identical splits within each target and protocol.',
        'Concentration grouped-condition folds separate companion technical measurements.',
        'Cumulative-release folds hold out complete pH/formulation profiles because neighbouring cumulative values share history.',
        'Forward-time evaluation trains exclusively on times earlier than the tested time.',
        'These are exploratory within-experiment tests, not independent batch validation.',
        'Zero-time values are included consistently. Their measured-versus-assumed origin is not inferred.',
        '',
        'Neural models use full-batch CPU Adam, the recorded fixed epoch budget, learning rate and weight decay.',
        'The MLP and GNN are newly implemented comparison models; old manuscript scores are not reused.',
        'The GNN has two dense graph-convolution blocks with 8 hidden features and a 16/8-unit regression head.',
        'Kinetic variants use two outputs for conditional capacity and rate; time is excluded from these parameter heads.',
        'Both formulations contain citral, so the same citral graph is used; formulation remains a separate input.',
        'Only one fixed molecule is present, so no topology generalisation, synergy or individual-compound importance is established.',
        'The unverified 13-compound extract dictionary is not treated as 13 measured payload-release experiments.',
        '',
        'GNN_kinetic and MLP_kinetic have the hard shape y=A(pH,form,chemistry)*(1-exp(-k(pH,form,chemistry)*t/96)).',
        'A is bounded to 0..100 percentage points and k is positive, giving y(0)=0 and a nondecreasing curve at fixed pH/formulation.',
        'They also use a soft empirical saturating-exponential regularizer fitted on training records only.',
        'This is a kinetically constrained surrogate, not a demonstrated PDE-informed or mechanistically validated model.',
        'No such cumulative prior is applied to instantaneous concentration.',
        'A kinetic fit requires two positive times; otherwise a nonnegative through-origin linear fallback is used.',
        'For a missing profile, classical fits use the same-pH other formulation, then pooled training profiles if necessary. Routes are saved.',
        'The kinetic variants enforce the stated shape; the other neural models do not. All extrapolation checks are reported without clipping.',
        'Observed non-monotonic release profiles are not altered to fit the constraint; residual errors remain visible.',
        '',
        'Configurations are fixed exploratory choices. No hyperparameter tuning uses the reported folds.',
        'Changing settings after inspecting these results constitutes exploratory development, not a fresh independent validation.',
        'R2 is omitted for constant observed targets. Inspect RMSE/MAE and subgroup metrics alongside pooled R2.',
        'RMSE/MAE for supplied release percentages have units of percentage points; concentration errors retain the supplied scale.',
        'Seed standard deviations describe computational variability, not uncertainty from biological experiments.',
        'Technical concentration SD is not assigned to cumulative means. No fake triplicates or fabricated release confidence intervals are produced.',
        'Full-data fits at pH 6.3 and 120 h are labelled unvalidated interpolation/extrapolation, not digital-twin validation.',
        'The 120-h neural output does not reproduce the manuscript bootstrap forecast and must not be presented as such.',
        '',
        '## Mean held-out performance across available seeds',
        comparison.to_csv(index=False),
    ]
    (output / 'analysis_notes.md').write_text('\n'.join(report), encoding='utf-8')
    write_json(output / 'completion.json', dict(status='completed', seconds=time.perf_counter() - start,
        held_out_predictions=len(results), source_sha256=hashlib.sha256(excel_path.read_bytes()).hexdigest()))
    print('COMPLETED. Results:', output)
    return dict(concentration=concentration, release=release, comparison=comparison,
                predictions=results, metrics=metrics, curves=curves, plots=plot_files)
