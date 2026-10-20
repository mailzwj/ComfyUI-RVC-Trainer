def main():
    from nodes_rvc_inference import (
        NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS,
        _voice_options, _index_options, _base_model_options,
    )

    expected = {"RVCVoiceLoader", "RVCHubertPitchLoader", "RVCExecute", "RVCAudioSynthesis"}
    assert expected == set(NODE_CLASS_MAPPINGS), NODE_CLASS_MAPPINGS
    assert expected == set(NODE_DISPLAY_NAME_MAPPINGS)

    fake = ["my-vocals.pth", "gangshou\\gangshou.pth", "pretrained_v2\\f0G48k.pth", "pretrained_v2\\D48k.pth"]
    voice = _voice_options(fake)
    assert voice == ["my-vocals.pth", "gangshou\\gangshou.pth"], voice

    fake_idx = ["sub\\m1.index", "sub\\m1.npy", "other.index"]
    assert _index_options(fake_idx) == ["(none)", "sub\\m1.index", "sub\\m1.npy", "other.index"]

    print("PASS: nodes register and option formatting")


if __name__ == "__main__":
    main()
