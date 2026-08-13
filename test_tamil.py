from pathlib import Path
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music

project_root = Path(__file__).parent

# Initialize DiT
print("Initializing DiT...")
dit_handler = AceStepHandler()
status, ok = dit_handler.initialize_service(
    project_root=str(project_root),
    config_path="acestep-v15-turbo",
    device="cuda",
    offload_to_cpu=True,
)
print(f"DiT: {status}")
if not ok:
    raise RuntimeError(f"DiT init failed: {status}")

# Initialize LLM
print("Initializing LLM...")
llm_handler = LLMHandler()
status, ok = llm_handler.initialize(
    checkpoint_dir=str(project_root / "checkpoints"),
    lm_model_path="acestep-5Hz-lm-1.7B",
    backend="vllm",
    device="cuda",
)
if not ok:
    print(f"LLM warning: {status}, continuing without LLM")
    llm_handler = None
else:
    print(f"LLM: {status}")

# Generate 15-second Tamil kuthu song
print("\nGenerating 15s Tamil kuthu song...")
params = GenerationParams(
    caption="energetic tamil kuthu dance song with clear tamil male vocals singing in tamil language, rich percussion, mridangam, dholak beats, synth bass, and catchy melody",
    lyrics="""[Verse]
ஆடுவோம் பாடுவோம் நண்பர்களே
இன்ப வாழ்வு வாழ்வோம் என்றும்
தாளத்தில் ஆடுவோம் ராகத்தில் பாடுவோம்
இசை மழை பொழியும் இன்று

[Chorus]
குத்து குத்து ஆடுவோம்
பீட்ஸ் ல நாடுவோம்
ரிதம் ல ஓடுவோம்
பார்ட்டி ல கூடுவோம்""",
    duration=60,
    vocal_language="ta",
    inference_steps=8,
    guidance_scale=7.0,
    thinking=True,
)

config = GenerationConfig(
    batch_size=1,
    audio_format="wav",
)

result = generate_music(
    dit_handler=dit_handler,
    llm_handler=llm_handler,
    params=params,
    config=config,
    save_dir=str(project_root / "output"),
)

print(f"\nSuccess: {result.success}")
print(f"Status: {result.status_message}")
if result.success:
    for audio in result.audios:
        print(f"Saved: {audio['path']}")
        print(f"Duration: {audio['tensor'].shape[-1] / audio['sample_rate']:.1f}s")
else:
    print(f"Error: {result.error}")
