import sys


def main():
    from rvc_inference import AVAILABLE_MODULES
    assert AVAILABLE_MODULES == ["rvc_inference.synths", "rvc_inference.loader", "rvc_inference.fcpe", "rvc_inference.index_loader", "rvc_inference.pipeline"], AVAILABLE_MODULES
    print("PASS: package imports")


if __name__ == "__main__":
    main()
