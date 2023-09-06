import os
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dir', required=True, help='output path')
    parser.add_argument('--name', required=True, help='name of experiment')
    args = parser.parse_args()
    min_epoch = -1
    for file in os.listdir(os.path.join(args.dir, args.name)):
        if file.endswith("_net.pth"):
            epoch = file.replace("_net.pth", "")
            if not epoch.isdigit():
                continue
            epoch = int(epoch)
            if epoch > min_epoch:
                min_epoch = epoch
    print(min_epoch)