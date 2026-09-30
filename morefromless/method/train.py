"""Run reference structured training with an explicitly provided adapter factory.

Factory(config) supplies policy, optimizer, batches, geometry and LossSettings.
Batches contain observation, demonstrated, valid_steps and training context.
No backbone/checkpoint, data split or experimental hyperparameter is inferred.
"""
import argparse
import importlib
import json
from pathlib import Path
import torch
from .structured import train_step


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--factory',required=True,help='Importable module:function adapter')
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--out-dir',type=Path,required=True)
    args=parser.parse_args()
    config=json.loads(args.config.read_text(encoding='utf-8'))
    required={'checkpoint_identity','dataset_manifest','epochs','seed','implementation_status'}
    if not required<=config.keys() or config['implementation_status']!='new_reference_run':
        raise ValueError("Explicit provenance, budget and new_reference_run status required")
    if not isinstance(config['epochs'],int) or config['epochs']<1:raise ValueError("Positive epoch budget required")
    module,name=args.factory.split(':',1)
    torch.manual_seed(config['seed'])
    adapter=getattr(importlib.import_module(module),name)(config)
    out=args.out_dir.resolve()
    if out.exists() and any(out.iterdir()):raise ValueError("Use a new empty output directory")
    out.mkdir(parents=True,exist_ok=True)
    (out/'run_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    adapter['policy'].train()
    with (out/'training.jsonl').open('w',encoding='utf-8') as log:
        for epoch in range(config['epochs']):
            count=0
            for batch in adapter['batches']():
                losses=train_step(adapter['policy'],adapter['optimizer'],geometry=adapter['geometry'],settings=adapter['settings'],**batch)
                log.write(json.dumps({'epoch':epoch,'batch':count,**losses})+'\n');count+=1
            if not count:raise ValueError("Empty training epoch")
    torch.save({'state_dict':adapter['policy'].state_dict(),'provenance':config},out/'reference_checkpoint.pt')

if __name__=='__main__':main()
