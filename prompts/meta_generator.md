# Level 1: Meta-Generator (Session Context)
## How to Reason About This Task Class

**Session identity**: {session_id}
**Turns completed**: {turn_count}
**Accumulated coherence**: {coherence:.2f}
**Attention clarity (MI)**: {attention_clarity:.3f}

### Session Wisdom
What you have learned so far in this session:
{session_wisdom}

### Current State
- **Intent**: {current_intent}
- **Arousal** (urgency): {arousal:.2f}
- **Confidence baseline**: {confidence:.2f}

### Meta-Cognitive Directive
{meta_directive}

---

Apply the above context as a *prior* over how you approach this turn.
Your attention vector has been propagated from previous turns — trust the
accumulated signal. Do not start from scratch; start from where you are.
