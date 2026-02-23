# Example: System Input and Output

This folder contains a real input/output pair from the JAADCR evaluation.

**Case**: Clear cell papulosis, April 2024, JAAD Case Reports (CC BY-NC-ND 4.0)
**Model**: MedGemma-1.5-4b-it (Vertex AI), with MCQ options format

---

## Files

| File | Description |
|------|-------------|
| `input_complete.txt` | Complete case: history + physical exam + histopathology. System produces a SOAP note. |
| `input_incomplete.txt` | History only: no physical exam, no image. System triggers agentic pause. |
| `output_soap.json` | System output for the complete case. Primary diagnosis: Clear cell papulosis (confidence 0.95). |
| `output_pause.json` | System output for the incomplete case. Agentic pause triggered, clarifying questions generated. |

---

## What This Demonstrates

**Complete input -> diagnosis:**
The system receives a full clinical picture (history, exam, histopathology, image) and produces a structured SOAP note with ranked differential diagnoses, confidence scores, and guideline citations.

**Incomplete input -> agentic pause:**
The same case with only patient history -- no physical exam, no image -- triggers the agentic pause. The system identifies the missing data, explains why it is clinically required, and generates targeted clarifying questions instead of guessing a diagnosis.
