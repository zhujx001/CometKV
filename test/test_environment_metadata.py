from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_default_requirements_exclude_vllm():
    requirements = (ROOT / "requirements.txt").read_text().splitlines()
    normalized = [line.strip().lower() for line in requirements]
    assert not any(line.startswith("vllm") for line in normalized)


def test_default_requirements_pin_transformers_5_stack():
    requirements = (ROOT / "requirements.txt").read_text().splitlines()
    normalized = [line.strip().lower() for line in requirements]
    assert "transformers==5.10.2" in normalized
    assert "huggingface-hub==1.17.0" in normalized
    assert "transformers==4.49.0" not in normalized
    assert "huggingface-hub==0.33.0" not in normalized


def test_default_requirements_include_direct_python_imports():
    requirements = (ROOT / "requirements.txt").read_text().splitlines()
    normalized = [line.strip().lower() for line in requirements]
    assert "numpy==2.2.6" in normalized
    assert "pyyaml==6.0.3" in normalized
    assert "nltk==3.9.4" in normalized
    assert "pandas==2.3.3" in normalized
    assert "pytest==9.0.3" in normalized
    assert "argparse" not in normalized


def test_cometkv_setup_does_not_hardcode_cuda_12_4():
    setup_py = (ROOT / "library" / "cometkv" / "setup.py").read_text()
    assert "/usr/local/cuda-12/include" not in setup_py
    assert "/usr/local/cuda-12.4" not in setup_py
    assert "CUDA_HOME" in setup_py


def test_cuda_128_requirements_use_prebuilt_flash_attention_wheel():
    requirements = (ROOT / "requirements-cu128.txt").read_text()
    assert "torch==2.8.0+cu128" in requirements
    assert "flashinfer-python==0.6.12" in requirements
    assert "flashinfer_jit_cache-0.6.12%2Bcu128" in requirements
    assert "flash_attn-2.8.3%2Bcu12torch2.8cxx11abiTRUE" in requirements
    assert "flash-attn==" not in requirements
    assert "--no-build-isolation" not in requirements
