# Change Log
### Version 1.0
#### Features
* Support for medical speech recognition with punctuatopm mission
#### Areas of improvement
* Design a **new preprocessing method** to Medical Speech Corpus in Chinese Medicine Speech Corpus (sChiMeS or psChiMeS with punctuation):
    
    * `chimes-14_data_prep.sh` in **espnet/egs/sChiMeS-14/asr1/local or espnet/egs/psChiMeS-14/asr1/local** divide audio files to two part (training set and testing set)
    * `text2tokenChimes.py` in **espnet/utils** changes English text to syllable formation and changes Chinese string to characters
    * `data2jsonChimes.sh` in **espnet/utils** change training data to json format

* Can train with or without **speed augmentation**.
* Also can train with **wave augmentation** using additional audio files.

* Evaluation code only for this model which trained on this corpus

---
