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

FIG = HERE.parent / "figures"       # the report's figures, drawn at print size: larger text on a slide
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
                        "2 backends × 8 configurations × 5 models (Qwen2.5-3B/7B/14B, Qwen3-8B/32B) on one GPU",
                        notes="Hardware, the eight configurations, the workloads and how we time things.")
    x0, w = 0.5, 2.9
    # platform card
    U.box(s, x0, U.TOP, w, U.BOT - U.TOP, fill=U.LIGHT, radius=0.08)
    U.card_header(s, x0, U.TOP, w, "LuMicrochip", "Platform")
    U.text(s, x0 + 0.16, U.TOP + 0.5, w - 0.3, 2.7, [
        [("NVIDIA RTX 5090", {"bold": True, "color": U.NAVY})],
        "Blackwell, 32 GB, 575 W",
        [("Measured: ", {"bold": True, "color": U.NAVY}), (f"{facts['bw']:.2f} TB/s, {facts['bf16']:.0f} TFLOP/s BF16"
                                                           + (f", {facts['fp8']:.0f} FP8" if facts.get('fp8') else ""), {})],
        [("Stacks: ", {"bold": True, "color": U.NAVY}), ("Transformers + bitsandbytes; vLLM", {})],
        [("Models: ", {"bold": True, "color": U.NAVY}), ("Qwen2.5-3B / 7B / 14B; Qwen3-8B, 32B", {})],
    ], size=11.5, space_after=6)
    # configs card (two columns)
    x1, w1 = x0 + w + 0.15, 9.5 - (x0 + w + 0.15)
    U.box(s, x1, U.TOP, w1, 1.85, fill=U.LIGHT, radius=0.08)
    U.card_header(s, x1, U.TOP, w1, "LuLayers", "Eight configurations")
    rows = [("Hugging Face + bitsandbytes", [("BF16", "2A78D6"), ("LLM.int8()", "D55181"), ("NF4", "C98500")]),
            ("vLLM + fused kernels", [("BF16", "2A78D6"), ("FP8", "008300"), ("GPTQ-INT8", "D55181"),
                                      ("AWQ / GPTQ-INT4", "C98500")])]
    y = U.TOP + 0.5
    for title, chips in rows:
        U.text(s, x1 + 0.16, y, w1 - 0.3, 0.2, [title], size=11, bold=True, color=U.GRAY)
        cx = x1 + 0.16
        for label, col in chips:
            wchip = 0.34 + 0.085 * len(label)
            U.box(s, cx, y + 0.24, wchip, 0.26, fill=U.WHITE, line=col, line_w=1.25, radius=0.06, decorative=False)
            d = U.oval(s, cx + 0.11, y + 0.37, 0.09, col)
            U.mark_decorative(d)
            U.text(s, cx + 0.2, y + 0.24, wchip - 0.2, 0.26, [label], size=11, color=U.TEXT, anchor="m")
            cx += wchip + 0.08
        y += 0.62
    # workload + metrics card
    y2 = U.TOP + 1.85 + 0.12
    U.box(s, x1, y2, w1, U.BOT - y2, fill=U.LIGHT, radius=0.08)
    U.card_header(s, x1, y2, w1, "LuGauge", "Workloads and metrics")
    U.text(s, x1 + 0.16, y2 + 0.48, w1 - 0.3, U.BOT - y2 - 0.5, [
        [("Sweep: ", {"bold": True, "color": U.NAVY}), ("batch 1–256 × prompt 128–2048 tokens", {})],
        [("Real traffic: ", {"bold": True, "color": U.NAVY}), ("300 ShareGPT requests, 1–64 clients", {})],
        [("Quality: ", {"bold": True, "color": U.NAVY}), ("WikiText-2, GSM8K, MMLU; paired tests", {})],
    ], size=11.5, space_after=3)


def slide_team_refs(prs):
    s = U.content_slide(prs, "Team and References", "Who did what, and what we built on",
                        notes="Division of work as planned in the proposal; everyone shared analysis and writing.")
    w = 2.9
    roles = [("Sijia Zhong", "Harness + memory (Q1)", "Harness, memory model, capacity", "LuMemoryStick"),
             ("Zheng Zheng", "Sweeps + kernels (Q2, Q3)", "Speed grids, kernels, profiling", "LuSlidersHorizontal"),
             ("Haofei Sun", "Traffic + quality (Q4)", "ShareGPT, WikiText-2, GSM8K, MMLU", "LuMessagesSquare")]
    for i, (name, role, task, ic) in enumerate(roles):
        x = 0.5 + i * (w + 0.15)
        U.box(s, x, U.TOP, w, 1.05, fill=U.LIGHT, radius=0.08)
        U.icon(s, x + 0.36, U.TOP + 0.36, 0.42, ic)
        U.text(s, x + 0.66, U.TOP + 0.12, w - 0.75, 0.24, [name], size=11.5, bold=True, color=U.NAVY, anchor="m")
        U.text(s, x + 0.66, U.TOP + 0.36, w - 0.75, 0.22, [role], size=11, color=U.BLUE_TXT, anchor="m")
        U.text(s, x + 0.16, U.TOP + 0.64, w - 0.3, 0.4, [task], size=10.5, color=U.TEXT)
    refs = [
        "Dettmers et al., LLM.int8(), NeurIPS 2022  ·  Dettmers et al., QLoRA (NF4), NeurIPS 2023",
        "Frantar et al., GPTQ, ICLR 2023  ·  Lin et al., AWQ, MLSys 2024  ·  Xiao et al., SmoothQuant, ICML 2023",
        "Frantar et al., MARLIN, PPoPP 2025  ·  Kwon et al., vLLM, SOSP 2023  ·  Lin et al., QServe, MLSys 2025",
        "Micikevicius et al., FP8 Formats, 2022  ·  Williams et al., Roofline, CACM 2009  ·  Kurtic et al., 2024",
        "Qwen2.5 / Qwen3 Technical Reports  ·  WikiText-2, GSM8K, MMLU benchmarks",
    ]
    y = U.TOP + 1.25
    U.card_header(s, 0.36, y - 0.1, 9.0, "LuBookOpen", "References")
    U.text(s, 0.5, y + 0.36, 9.0, 1.6, refs, size=10, color=U.TEXT, space_after=3)
    U.text(s, 5.2, 4.72, 4.3, 0.5, [[("Code, data and report: ", {"bold": True, "color": U.NAVY}),
                                     ("github.com/zz9tf/6359-SYS-ARCHITECT-course-work", {})]], size=10, color=U.GRAY)


def main():
    from quantbench import analysis as A
    peaks = A.hw_peaks()
    facts = {"bw": peaks.get("dram_read_gbps", 0) / 1e3, "bf16": peaks.get("bf16_gemm_tflops", 0),
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
