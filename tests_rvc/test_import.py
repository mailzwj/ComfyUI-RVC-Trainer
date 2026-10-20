import sys


def main():
    from rvc_inference import AVAILABLE_MODULES
    assert AVAILABLE_MODULES == ["rvc_inference.synths"], AVAILABLE_MODULES
    print("PASS: package imports")


if __name__ == "__main__":
    main()
