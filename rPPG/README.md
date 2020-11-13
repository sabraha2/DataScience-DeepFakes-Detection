## Pulse Features for Deepfakes Detection
The general flow for extracting the rPPG features is to first extract the face landmarks from the Preprocessing directory, then perform the actual pulse detection, followed by decomposing the pulse signal into its spectral components.

In order, do:
(1) extract_pulse.py
(2) extract_spectral_features.py
(3) classifier_predictions.py
(4) evaluate_predictions.py
