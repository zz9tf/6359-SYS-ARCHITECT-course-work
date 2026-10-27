#!/usr/bin/env python
"""Build the final-presentation deck (UTA template) from the figures and results of the study.

    python slides/build_slides.py            # -> slides/final_presentation.pptx
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import uta as U  # noqa: E402
from results_slides import add_results_slides  # noqa: E402

FIG = HERE.parent / "figures" / "slides"
OUT = HERE / "final_presentation.pptx"


def slide_title(prs):
    s = prs.slides.add_slide(prs.slide_layouts[0])
    U.place(s.shapes.title, 0.669, 1.62, 9.0, 0.8)
    U.set_ph(s.shapes.title, "Smaller Is Not Always Faster", size=40)
    U.place(s.placeholders[10], 0.669, 2.36, 9.0, 0.47)
    U.set_ph(s.placeholders[10], "LLM Quantization on an RTX 5090: Memory vs. Speed vs. Quality")
    U.place(s.placeholders[11], 0.669, 2.998, 8.0, 0.33)
    U.set_ph(s.placeholders[11], "Sijia Zhong  ·  Zheng Zheng  ·  Haofei Sun")
    U.place(s.placeholders[12], 0.669, 3.33, 8.0, 0.32)
    U.set_ph(s.placeholders[12], "Final Project  ·  Fall 2026")
    U.text(s, 0.76, 1.3, 8.0, 0.3, [[("CSE 6359", {"bold": True}),
                                     ("  ·  Advanced Topics in Systems & Architecture", {})]],
           size=12, color=U.WHITE, anchor="m")
    s.notes_slide.notes_text_frame.text = (
        "We studied what quantization really buys you when serving an LLM: memory, speed and quality, "
        "measured end to end on one RTX 5090 with two inference stacks.")


def slide_setup(prs, facts):
    s = U.content_slide(prs, "What We Measured",
                        "2 backends × 8 configurations × 3 model sizes — every number measured on one GPU",
                        notes="Hardware, the eight configurations, the workloads and how we time things.")
    x0, w = 0.5, 2.9
    # platform card
    U.box(s, x0, U.TOP, w, U.BOT - U.TOP, fill=U.LIGHT, radius=0.08)
    U.card_header(s, x0, U.TOP, w, "LuMicrochip", "Platform")
    U.text(s, x0 + 0.16, U.TOP + 0.5, w - 0.3, 2.7, [
        [("NVIDIA RTX 5090", {"bold": True, "color": U.NAVY}), (" (Blackwell, sm_120)", {})],
        "32 GB GDDR7 · 170 SMs · 575 W",
        [("Measured: ", {"bold": True, "color": U.NAVY}), (f"{facts['bw']:.2f} TB/s DRAM, {facts['bf16']:.0f} TFLOP/s BF16"
                                                           + (f", {facts['fp8']:.0f} TFLOP/s FP8" if facts.get('fp8') else ""), {})],
        [("Software: ", {"bold": True, "color": U.NAVY}), ("PyTorch 2.13, Transformers 5.19, bitsandbytes 0.50, vLLM 0.32", {})],
        [("Models: ", {"bold": True, "color": U.NAVY}), ("Qwen2.5-3B / 7B / 14B-Instruct", {})],
    ], size=9.5, space_after=4)
    # configs card (two columns)
    x1, w1 = x0 + w + 0.15, 9.5 - (x0 + w + 0.15)
    U.box(s, x1, U.TOP, w1, 1.85, fill=U.LIGHT, radius=0.08)
    U.card_header(s, x1, U.TOP, w1, "LuLayers", "Eight configurations")
    rows = [("Hugging Face (eager) + bitsandbytes, quantized at load", [("BF16", "2A78D6"), ("INT8  LLM.int8()", "D55181"), ("NF4  4-bit", "C98500")]),
            ("vLLM (CUDA graphs) + fused kernels, official checkpoints", [("BF16", "2A78D6"), ("FP8  W8A8", "008300"), ("INT8  GPTQ W8A16", "D55181"),
                                                                          ("INT4  AWQ / GPTQ W4A16", "C98500")])]
    y = U.TOP + 0.5
    for title, chips in rows:
        U.text(s, x1 + 0.16, y, w1 - 0.3, 0.2, [title], size=8.5, bold=True, color=U.GRAY)
        cx = x1 + 0.16
        for label, col in chips:
            wchip = 0.3 + 0.066 * len(label)
            U.box(s, cx, y + 0.24, wchip, 0.26, fill=U.WHITE, line=col, line_w=1.25, radius=0.06, decorative=False)
            d = U.oval(s, cx + 0.11, y + 0.37, 0.09, col)
            U.mark_decorative(d)
            U.text(s, cx + 0.2, y + 0.24, wchip - 0.2, 0.26, [label], size=8, color=U.TEXT, anchor="m")
            cx += wchip + 0.08
        y += 0.62
    # workload + metrics card
    y2 = U.TOP + 1.85 + 0.12
    U.box(s, x1, y2, w1, U.BOT - y2, fill=U.LIGHT, radius=0.08)
    U.card_header(s, x1, y2, w1, "LuGauge", "Workloads and metrics")
    U.text(s, x1 + 0.16, y2 + 0.48, w1 - 0.3, U.BOT - y2 - 0.5, [
        [("Controlled sweep: ", {"bold": True, "color": U.NAVY}),
         ("batch 1–256 × prompt 128–2048 tokens, exactly 128 output tokens", {})],
        [("Real workload: ", {"bold": True, "color": U.NAVY}),
         ("300 ShareGPT requests (fixed seed), 1–64 concurrent clients", {})],
        [("Quality: ", {"bold": True, "color": U.NAVY}), ("WikiText-2 perplexity, GSM8K (all 1,319 questions)", {})],
        [("Metrics: ", {"bold": True, "color": U.NAVY}),
         ("peak memory, TTFT, TPOT, tokens/s, energy (J/token); 1 warm-up + 3 repeats", {})],
    ], size=9, space_after=3)


def slide_team_refs(prs):
    s = U.content_slide(prs, "Team and References", "Who did what, and what we built on",
                        notes="Division of work as planned in the proposal; everyone shared analysis and writing.")
    w = 2.9
    roles = [("Sijia Zhong", "Baseline + memory (Q1)", "Benchmark harness, memory model, capacity", "LuMemoryStick"),
             ("Zheng Zheng", "Controlled sweeps (Q2)", "Precision × batch × length grids, kernels, profiling", "LuSlidersHorizontal"),
             ("Haofei Sun", "Real workload + quality (Q3, Q4)", "ShareGPT serving, WikiText-2, GSM8K", "LuMessagesSquare")]
    for i, (name, role, task, ic) in enumerate(roles):
        x = 0.5 + i * (w + 0.15)
        U.box(s, x, U.TOP, w, 1.05, fill=U.LIGHT, radius=0.08)
        U.icon(s, x + 0.36, U.TOP + 0.36, 0.42, ic)
        U.text(s, x + 0.66, U.TOP + 0.12, w - 0.75, 0.24, [name], size=11.5, bold=True, color=U.NAVY, anchor="m")
        U.text(s, x + 0.66, U.TOP + 0.36, w - 0.75, 0.22, [role], size=9, color=U.BLUE_TXT, anchor="m")
        U.text(s, x + 0.16, U.TOP + 0.66, w - 0.3, 0.36, [task], size=8.5, color=U.TEXT)
    refs = [
        "Dettmers et al., LLM.int8(), NeurIPS 2022  ·  Dettmers et al., QLoRA (NF4), NeurIPS 2023",
        "Frantar et al., GPTQ, ICLR 2023  ·  Lin et al., AWQ, MLSys 2024  ·  Xiao et al., SmoothQuant, ICML 2023",
        "Frantar et al., MARLIN mixed-precision kernels, PPoPP 2025  ·  Kwon et al., vLLM / PagedAttention, SOSP 2023",
        "Micikevicius et al., FP8 Formats for Deep Learning, 2022  ·  Williams et al., Roofline, CACM 2009",
        "Qwen Team, Qwen2.5 Technical Report, 2024  ·  Merity et al., WikiText (ICLR 2017)  ·  Cobbe et al., GSM8K, 2021",
    ]
    y = U.TOP + 1.25
    U.card_header(s, 0.36, y - 0.1, 9.0, "LuBookOpen", "References")
    U.text(s, 0.5, y + 0.36, 9.0, 1.6, refs, size=8.5, color=U.TEXT, space_after=4)
    U.text(s, 6.0, 4.75, 3.5, 0.5, [[("Code, data and report: ", {"bold": True, "color": U.NAVY}),
                                     ("github.com/zz9tf/6359-SYS-ARCHITECT-course-work", {})]], size=8, color=U.GRAY)


def main():
    from quantbench import analysis as A
    peaks = A.hw_peaks()
    facts = {"bw": peaks.get("dram_copy_gbps", 0) / 1e3, "bf16": peaks.get("bf16_gemm_tflops", 0),
             "fp8": peaks.get("fp8_gemm_tflops")}
    prs = U.template()
    slide_title(prs)
    add_results_slides(prs, FIG, facts, setup=lambda: slide_setup(prs, facts))
    slide_team_refs(prs)
    cp = prs.core_properties
    cp.title = "Smaller Is Not Always Faster: LLM Quantization on an RTX 5090"
    cp.author = "Sijia Zhong, Zheng Zheng, Haofei Sun"
    cp.last_modified_by = "Sijia Zhong, Zheng Zheng, Haofei Sun"
    cp.subject = "CSE 6359 final project"
    prs.save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
