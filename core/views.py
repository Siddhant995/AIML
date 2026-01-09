import os
import json
import pandas as pd
import numpy as np
from django.shortcuts import render, redirect
from django.conf import settings
from django.http import JsonResponse, HttpResponse
from .automl import run_automl
UPLOAD_FOLDER = os.path.join(settings.BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def home(request):
    return render(request, 'home.html')

def upload_file(request):
    if request.method == 'POST' and request.FILES.get('file'):
        f = request.FILES['file']
        fname = f.name
        path = os.path.join(UPLOAD_FOLDER, fname)
        with open(path, 'wb') as dest:
            for chunk in f.chunks():
                dest.write(chunk)
        # save filename in session
        request.session['current_file'] = fname
        return redirect('overview')
    return redirect('home')

def _load_df(request):
    fname = request.session.get('current_file')
    if not fname:
        return None
    path = os.path.join(UPLOAD_FOLDER, fname)
    if not os.path.exists(path):
        return None
    if fname.lower().endswith('.csv'):
        return pd.read_csv(path)
    else:
        return pd.read_excel(path)

def overview(request):
    df = _load_df(request)
    if df is None:
        return redirect('home')
    desc_html = df.describe(include='all').to_html(classes='table table-sm', border=0)
    total_cells = df.size
    nulls = int(df.isnull().sum().sum())
    null_pct = round(100 * nulls / total_cells, 3) if total_cells>0 else 0
    dupes = int(df.duplicated().sum())
    dupes_pct = round(100 * dupes / len(df), 3) if len(df)>0 else 0
    fields = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        types = set([type(x).__name__ for x in df[col].dropna().map(lambda v: type(v)).unique()]) if len(df[col].dropna())>0 else set()
        if len(types) > 1:
            dtype = 'mixed'
        fields.append({'col': col, 'dtype': dtype})
    context = {'desc_html': desc_html, 'nulls': nulls, 'null_pct': null_pct,
               'dupes': dupes, 'dupes_pct': dupes_pct, 'fields': fields}
    return render(request, 'overview.html', context)

def column_view(request, col):
    df = _load_df(request)
    if df is None or col not in df.columns:
        return redirect('overview')
    col_series = df[col]
    nulls = int(col_series.isnull().sum())
    dupes = int(col_series.duplicated().sum())
    unique_vals = None
    if request.method == 'POST':
        action = request.POST.get('action')
        if action == 'unique':
            unique_vals = [str(v) for v in col_series.dropna().unique().tolist()]
            #unique_vals = list(col_series.dropna().unique())
        elif action == 'fill_zero':
            df[col] = col_series.fillna(0)
            df.to_csv(os.path.join(UPLOAD_FOLDER, request.session['current_file']), index=False)
        elif action == 'fill_mean':
            if pd.api.types.is_numeric_dtype(col_series):
                df[col] = col_series.fillna(col_series.mean())
                df.to_csv(os.path.join(UPLOAD_FOLDER, request.session['current_file']), index=False)
        elif action == 'fill_median':
            if pd.api.types.is_numeric_dtype(col_series):
                df[col] = col_series.fillna(col_series.median())
                df.to_csv(os.path.join(UPLOAD_FOLDER, request.session['current_file']), index=False)
        elif action == 'fill_mode':
            df[col] = col_series.fillna(col_series.mode().iloc[0] if not col_series.mode().empty else 0)
            df.to_csv(os.path.join(UPLOAD_FOLDER, request.session['current_file']), index=False)
        # reload after changes
        df = _load_df(request)
        col_series = df[col]
        nulls = int(col_series.isnull().sum())
        dupes = int(col_series.duplicated().sum())
    return render(request, 'column.html', {'col': col, 'nulls': nulls, 'dupes': dupes, 'unique_vals': unique_vals})

def predict_select(request):
    df = _load_df(request)
    if df is None:
        return redirect('home')
    cols = list(df.columns)
    return render(request, 'predict_select.html', {'fields': cols})

def predict_run(request):
    df = _load_df(request)
    if df is None:
        return redirect('home')
    if request.method == 'POST':
        selected = request.POST.getlist('targets')
        if not selected:
            return HttpResponse('select at least one target', status=400)
        if len(selected) > 1:
            return HttpResponse('multi-target not implemented', status=400)
        automl_res = run_automl(df, selected[0])
        return render(request, 'predict.html', {'automl_res': automl_res})
    return redirect('predict_select')

def visualize(request):
    df = _load_df(request)
    if df is None:
        return redirect('home')
    cols = list(df.columns)
    if request.method == 'POST':
        x = request.POST.get('x_col')
        y = request.POST.get('y_col')
        if x and y:
            plot_json = df[[x,y]].dropna().to_dict(orient='list')
            return JsonResponse({'x': plot_json[x], 'y': plot_json[y], 'x_label': x, 'y_label': y})
    return render(request, 'visualize.html', {'fields': cols})
