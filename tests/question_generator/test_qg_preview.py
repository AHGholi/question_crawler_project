# tests/question_generator/test_qg_preview.py
from question_generator.local_hf_qg import LocalHFQuestionGenerator
from question_generator.base import QGInput

def test_preview_questions():
    qg = LocalHFQuestionGenerator(model="distilgpt2", do_sample=True, temperature=0.7)
    data = QGInput(
        topic="Photosynthesis",
        title="How plants make food",
        summary="Plants convert light energy into chemical energy in chloroplasts.",
        keywords=["chlorophyll", "light-dependent reactions", "Calvin cycle", "glucose", "stomata"],
        sentences=[
            "Photosynthesis occurs mainly in leaf mesophyll cells.",
            "Chlorophyll absorbs blue and red wavelengths of light.",
            "ATP and NADPH are produced in the light-dependent reactions.",
            "The Calvin cycle fixes carbon dioxide to synthesize sugars.",
        ],
        num_questions=5,
    )
    qs = qg.generate(data)
    print("\nGenerated questions:")
    for i, q in enumerate(qs, 1):
        print(f"{i}. {q}")

    assert len(qs) > 0
