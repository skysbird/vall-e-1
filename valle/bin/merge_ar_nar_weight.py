"""
since we train ar and nar stage separately for efficient consideration, now we need to merge weights of these two parts into one
"""

import torch
import argparse


def merge(ar,nar,output):
    ar_weight_pt = ar
    nar_weight_pt = nar 
    
    ar_ckpt = torch.load(ar_weight_pt, map_location="cpu")
    nar_ckpt = torch.load(nar_weight_pt, map_location="cpu")

    print(ar_ckpt.keys())
    print(nar_ckpt.keys())

    #assert set(ar_ckpt.keys()) ==  set(nar_ckpt.keys())
    
    
    for key in ar_ckpt["model"].keys():
        if "nar" in key:
            ar_ckpt["model"][key] = nar_ckpt["model"][key]
    
    torch.save(ar_ckpt, output)



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ar",
        type=str,
        default="",
        help="Ar model",
    )


    parser.add_argument(
        "--nar",
        type=str,
        default="",
        help="nar model",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="output-m.pt",
        help="",
    )

    return parser.parse_args()


@torch.no_grad()
def main():
    args = get_args()
    merge(args.ar, args.nar, args.output)

if __name__ == "__main__":
    main()


