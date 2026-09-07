# Fixed sentence set — read the same 10 sentences in every recording

**Translation caveat, stated honestly:** the Hindi and Marathi text below
was drafted for structural/thematic coverage (urgency, financial requests,
claimed authority — the actual attack pattern this project defends
against), not verified by a native speaker. **A native Hindi/Marathi
speaker on the team must review and correct these before recording any
real volunteer** — don't record from this file as-is without that review.
Malvi has no entries here at all; it needs a native speaker to draft from
scratch, there is no machine-translation starting point worth trusting for
an unwritten dialect.

Each sentence is read genuinely (as the volunteer, in their own voice) and
— separately, only with the consent in `../CONSENT_FORM.md` — the same
text is later fed to a synthesis system using the volunteer's genuine
recording as a reference, so genuine and synthetic pairs share content.

| # | English (source) | Hindi (needs native review) | Marathi (needs native review) |
|---|---|---|---|
| 1 | Hello, this is Rajesh calling from the finance department. | नमस्ते, मैं राजेश बोल रहा हूँ, वित्त विभाग से। | नमस्कार, मी राजेश बोलतोय, वित्त विभागातून। |
| 2 | I need you to transfer the funds immediately, it's urgent. | मुझे चाहिए कि आप तुरंत पैसे ट्रांसफर करें, यह अत्यावश्यक है। | मला हवे आहे की तुम्ही लगेच पैसे ट्रान्सफर करा, हे खूप तातडीचे आहे। |
| 3 | Please don't tell anyone about this call yet. | कृपया अभी इस कॉल के बारे में किसी को न बताएं। | कृपया सध्या या कॉलबद्दल कोणालाही सांगू नका। |
| 4 | The account details will be sent to you right after this call. | खाते का विवरण इस कॉल के तुरंत बाद आपको भेज दिया जाएगा। | खात्याचे तपशील या कॉलनंतर लगेच तुम्हाला पाठवले जातील। |
| 5 | I'm currently in a meeting and cannot verify this myself. | मैं अभी एक बैठक में हूँ और इसे स्वयं सत्यापित नहीं कर सकता। | मी सध्या बैठकीत आहे आणि हे स्वतः पडताळू शकत नाही। |
| 6 | This transaction needs to be completed before the end of the day. | यह लेन-देन आज दिन खत्म होने से पहले पूरा होना चाहिए। | हा व्यवहार आजचा दिवस संपण्यापूर्वी पूर्ण झाला पाहिजे। |
| 7 | Can you confirm the last four digits of the account number? | क्या आप खाता संख्या के अंतिम चार अंकों की पुष्टि कर सकते हैं? | तुम्ही खाते क्रमांकाचे शेवटचे चार अंक निश्चित करू शकाल का? |
| 8 | I understand this is unusual, but please trust me on this. | मुझे पता है यह असामान्य है, लेकिन कृपया इस पर मुझ पर भरोसा करें। | मला माहीत आहे हे असामान्य आहे, पण कृपया यावर माझ्यावर विश्वास ठेवा। |
| 9 | Thank you for your help, I'll call you back once it's done. | आपकी मदद के लिए धन्यवाद, यह हो जाने पर मैं आपको वापस कॉल करूँगा। | तुमच्या मदतीबद्दल धन्यवाद, हे झाल्यावर मी तुम्हाला परत कॉल करेन। |
| 10 | Good morning, how are you doing today? | सुप्रभात, आज आप कैसे हैं? | सुप्रभात, आज तुम्ही कसे आहात? |

Sentence 10 is a deliberately neutral baseline (no urgency/financial
framing) — included so the dataset isn't *entirely* pressure-themed
sentences, which would confound "is this synthetic" with "is this an
urgent-sounding sentence."
