"""Run the data analyses and optional simulations supplied with this study."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys
import importlib

ROOT = Path(__file__).resolve().parent
TASKS = ['surrogate','morris','sobol','ranking','scan','optimization','uncertainty','mesh']


def verify(root):
    """Verify the published code, configuration, models and data without dependencies."""
    manifest = root/'CHECKSUMS.csv'
    if not manifest.is_file():raise ValueError('CHECKSUMS.csv is missing.')
    checked=0
    with manifest.open(encoding='utf-8-sig',newline='') as handle:
        for row in csv.DictReader(handle):
            path=(root/row['path']).resolve()
            if not path.is_relative_to(root) or not path.is_file():raise ValueError(f'Missing or invalid file: {row["path"]}')
            h=hashlib.sha256()
            with path.open('rb') as data:
                for block in iter(lambda:data.read(1024*1024),b''):h.update(block)
            if path.stat().st_size!=int(row['bytes']) or h.hexdigest()!=row['sha256']:
                raise ValueError(f'Checksum mismatch: {row["path"]}')
            checked+=1
    return {'verified_files':checked,'status':'PASS','scope':'Published source and data. Generated analysis outputs are not part of the input checksums.'}


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT,help='Project directory (default: this script directory)')
    p.add_argument('--out',type=Path,help='Output directory (default: PROJECT/results)')
    subs=p.add_subparsers(dest='command')
    subs.add_parser('verify',help='Check supplied files using only the Python standard library')
    a=subs.add_parser('analyze',help='Recalculate results from the supplied data without COMSOL')
    a.add_argument('--task',choices=['all']+TASKS,default='all')
    a.add_argument('--bootstrap',type=int,default=20000,help='Trajectory resamples for Morris')
    f=subs.add_parser('fit',help='Refit the Gaussian process from the supplied training CSV')
    o=subs.add_parser('optimize',help='Run a new NSGA-II search with the declared parameters')
    o.add_argument('--seed',type=int,nargs='+',default=[20260816]);o.add_argument('--generations',type=int,default=200)
    o.add_argument('--scenario',default='baseline',help='Scenario name from config/optimization.json')
    m=subs.add_parser('sample',help='Generate a new conditional Monte Carlo draw table')
    m.add_argument('--seed',type=int,default=20260815);m.add_argument('--samples',type=int,default=2000)
    c=subs.add_parser('comsol',help='Prepare Java jobs; optionally compile and run them with COMSOL')
    c.add_argument('--study',choices=['local-study','export-fields','morris','training'],required=True)
    c.add_argument('--comsol-root',type=Path);c.add_argument('--execute',action='store_true')
    c.add_argument('--seed',type=int,nargs='+');c.add_argument('--level',choices=['L01','L02','L03'],nargs='+');c.add_argument('--case',choices=['sharp_rect','rounded_R2'],nargs='+')
    for command_parser in [a,f,o,m,c]:
        command_parser.add_argument('--out',type=Path,default=argparse.SUPPRESS,help='Output directory')
    return p


def main(argv=None):
    args=parser().parse_args(argv);root=args.root.resolve();out=(args.out or root/'results').resolve()
    command=args.command or 'analyze'
    if command != 'verify':
        if out == root or any(out == root/name or (root/name) in out.parents
                              for name in ('data','models','config','src','docs')):
            raise ValueError('Choose an output directory outside the supplied source, data and model directories.')
    if command=='verify':result=verify(root)
    elif command=='analyze':
        bootstrap=getattr(args,'bootstrap',20000)
        if bootstrap<1:raise ValueError('bootstrap must be positive.')
        runners={
            'surrogate':('surrogate','run_surrogate',out,{}),
            'morris':('morris','run_morris',out/'morris',{'bootstrap':bootstrap}),
            'sobol':('sensitivity','run_sensitivity',out,{}),
            'ranking':('ranking','run_ranking',out,{}),
            'scan':('ranking','run_scan',out,{}),
            'optimization':('optimization','run_optimization_analysis',out/'optimization',{}),
            'uncertainty':('uncertainty','run_uncertainty',out,{}),
            'mesh':('comsol','run_mesh_analysis',out/'mesh',{})}
        selected=TASKS if getattr(args,'task','all')=='all' else [args.task]
        result={}
        for task in selected:
            print(f'Analyzing {task} ...',flush=True)
            module,name,destination,options=runners[task]
            function=getattr(importlib.import_module('src.'+module),name)
            result[task]=function(root,destination,**options)
        out.mkdir(parents=True,exist_ok=True)
        (out/'summary.json').write_text(json.dumps(result,indent=2,default=str,allow_nan=False)+'\n',encoding='utf-8')
    elif command=='fit':
        from src.surrogate import run_surrogate
        result=run_surrogate(root,out,refit=True)
    elif command=='optimize':
        from src.optimization import reoptimize
        if args.generations<1:raise ValueError('generations must be positive.')
        result=reoptimize(root,out,seeds=args.seed,generations=args.generations,scenario=args.scenario)
    elif command=='sample':
        from src.uncertainty import regenerate_samples
        result=regenerate_samples(root,out,samples_per_candidate=args.samples,seed=args.seed)
    elif command=='comsol':
        from src.comsol import prepare_comsol
        result=prepare_comsol(root,out,args.study,comsol_root=args.comsol_root,execute=args.execute,seeds=args.seed,levels=args.level,cases=args.case)
    else:raise ValueError(f'Unknown command: {command}')
    print(json.dumps(result,indent=2,default=str))
    return result


if __name__=='__main__':
    try:main()
    except (ValueError,FileNotFoundError,ImportError,RuntimeError) as error:
        print(f'Error: {error}',file=sys.stderr);sys.exit(1)
