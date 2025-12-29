import argparse
import os

from training import train, test

def main(mode=None, config_path=None, ckpt=None, gpu_id="0"):

    # 使用 argparse.ArgumentParser
    ap = argparse.ArgumentParser("Sign-KID")

    # 如果没有在函数参数中传递 mode 和 config_path，则使用命令行参数
    if mode is None or config_path is None:
        ap.add_argument("mode", choices=["train", "test"], help="train a model or test")
        ap.add_argument("config_path", type=str, help="path to YAML config file")

    # 其他可选参数
    ap.add_argument("--ckpt", type=str, help="path to model checkpoint")
    ap.add_argument("--gpu_id", type=str, default=gpu_id, help="gpu to run your job on")

    # 如果通过函数参数传递了 mode、config_path 和 ckpt，则使用它们
    if mode is not None and config_path is not None:
        args = argparse.Namespace(mode=mode, config_path=config_path, ckpt=ckpt, gpu_id=gpu_id)
    else:
        # 否则，通过命令行解析参数
        args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # 根据 mode 决定调用 train 或 test
    if args.mode == "train":
        train(cfg_file=args.config_path, ckpt=args.ckpt)
    elif args.mode == "test":
        test(cfg_file=args.config_path, ckpt=args.ckpt)
    else:
        raise ValueError("Unknown mode")

if __name__ == "__main__":
    main()
