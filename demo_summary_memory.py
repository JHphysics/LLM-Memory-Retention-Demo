#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import random
import re
from textwrap import dedent
import matplotlib.pyplot as plt

MODEL_NAME = "llama3"
TEMPERATURE = 0
TOTAL_TURNS = 50
TEST_EVERY = 5

# 메모리 관련 설정
RECENT_WINDOW_SIZE = 8       # 최근 몇 개 대화 turn을 raw로 유지할지
SUMMARY_UPDATE_EVERY = 3     # 몇 turn마다 summary를 갱신할지

llm = ChatOllama(model=MODEL_NAME, temperature=TEMPERATURE)

# -----------------------------------
# 1. 기억해야 할 "복합 프로필"
# -----------------------------------
TRUE_PROFILE = {
    "residence": "Busan Haeundae",
    "cat_name": "Mango",
    "cat_adopt_year": "2021",
    "preferred_evening_exercise": "Swimming",
    "backup_exercise_when_knee_hurts": "Walking",
    "favorite_drink": "Americano",
    "avoid_caffeine_after": "7 PM",
}

PROFILE_TEXT = dedent(f"""
Please remember the following user profile carefully.

- The user currently lives near {TRUE_PROFILE['residence']}.
- The user's cat is named {TRUE_PROFILE['cat_name']}, and the cat was adopted in {TRUE_PROFILE['cat_adopt_year']}.
- The user's preferred evening exercise is {TRUE_PROFILE['preferred_evening_exercise']}.
- When the user's knee hurts, the user chooses {TRUE_PROFILE['backup_exercise_when_knee_hurts']} instead.
- The user's favorite drink is {TRUE_PROFILE['favorite_drink']}, but the user avoids caffeine after {TRUE_PROFILE['avoid_caffeine_after']}.

Please confirm briefly.
""").strip()

# -----------------------------------
# 2. 혼동 정보
# -----------------------------------
DISTRACTORS = [
    "My friend lives in Incheon Songdo.",
    "My cousin's cat is named Coco and was adopted in 2020.",
    "My brother prefers running in the evening.",
    "When my ankle hurts, I stretch instead of jogging.",
    "My coworker drinks latte every afternoon.",
    "I used to live in Daegu before moving.",
    "My sister avoids caffeine after 9 PM.",
    "A friend likes tennis after work.",
    "Another pet name I heard recently was Nabi.",
    "Someone I know switched from Americano to Mocha.",
]

LONG_NOISE = [
    """This paragraph is intentionally long and distracting. It includes many entities such as names,
    places, time references, beverages, and activities. The purpose is to increase context load and
    make retrieval of earlier facts more difficult. Long dialogue often causes confusion between the
    user's own facts and facts that belong to other people.""",

    """In extended conversations, models may confuse who owns which attribute. For example, a city
    may belong to a friend, a drink preference may belong to a coworker, and a pet name may belong
    to a cousin. This kind of interference is useful for stress-testing raw-history dialogue systems.""",

    """Many similar facts can coexist in a single context: exercise preferences, injury conditions,
    locations, schedules, adoption years, and drink choices. The more overlap there is, the harder
    it becomes to cleanly retrieve only the target user's information.""",
]

# -----------------------------------
# 3. 다양한 질문 세트
# -----------------------------------
QUESTION_BANK = [
    {
        "name": "residence_only",
        "question": "Where does the user currently live?",
        "targets": ["Busan Haeundae"]
    },
    {
        "name": "cat_name_and_year",
        "question": "Tell me the user's cat name and adoption year together.",
        "targets": ["Mango", "2021"]
    },
    {
        "name": "exercise_preference",
        "question": "What exercise does the user prefer in the evening?",
        "targets": ["Swimming"]
    },
    {
        "name": "knee_condition",
        "question": "When the user's knee hurts, what does the user choose instead?",
        "targets": ["Walking"]
    },
    {
        "name": "drink_rule",
        "question": "What does the user drink most, and after what time does the user avoid caffeine?",
        "targets": ["Americano", "7 PM"]
    },
    {
        "name": "profile_summary",
        "question": "Summarize only the user's profile in 3 short bullet points. Do not include other people's information.",
        "targets": ["Busan Haeundae", "Mango", "2021", "Swimming", "Walking", "Americano", "7 PM"]
    },
]

# -----------------------------------
# 4. 문자열 정규화
# -----------------------------------
def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9가-힣\s:.-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

def count_target_hits(answer, targets):
    answer_n = normalize_text(answer)
    hits = 0
    for t in targets:
        if normalize_text(t) in answer_n:
            hits += 1
    return hits

def score_item(answer, targets):
    hits = count_target_hits(answer, targets)
    return hits / len(targets)

# -----------------------------------
# 5. 질문 표현 바꾸기
# -----------------------------------
PARAPHRASE_MAP = {
    "residence_only": [
        "What area does the user currently live near?",
        "Can you recall the user's current residential area?",
        "Where is the user's present home location?"
    ],
    "cat_name_and_year": [
        "What is the name of the user's cat, and when was it adopted?",
        "Recall the pet's name and adoption year.",
        "Tell me the cat information: name plus adoption year."
    ],
    "exercise_preference": [
        "Which workout does the user usually prefer after work?",
        "What is the user's favorite evening exercise?",
        "What activity does the user mainly choose at night?"
    ],
    "knee_condition": [
        "If the user's knee is uncomfortable, what is the fallback exercise?",
        "What does the user do instead when knee pain appears?",
        "Which activity replaces the usual one when the knee hurts?"
    ],
    "drink_rule": [
        "What beverage does the user prefer, and when is caffeine avoided?",
        "Tell me the user's drink preference and caffeine cutoff time.",
        "What is the user's usual coffee choice, and after what hour is caffeine avoided?"
    ],
    "profile_summary": [
        "List only the user's own profile facts in bullets.",
        "Give a short profile of the user without mixing in others.",
        "Summarize the user's identity-related facts only."
    ],
}

# -----------------------------------
# 6. Summary Memory 모듈
# -----------------------------------
class SummaryMemory:
    def __init__(self, llm, recent_window_size=8):
        self.llm = llm
        self.recent_window_size = recent_window_size
        self.full_history = []         # (role, text)
        self.summary = ""              # 압축된 장기 기억
        self.last_summarized_idx = 0   # 어디까지 summary에 반영했는지 추적

    def add_user(self, text):
        self.full_history.append(("user", text))

    def add_ai(self, text):
        self.full_history.append(("assistant", text))

    def _format_history_slice(self, history_slice):
        lines = []
        for role, text in history_slice:
            prefix = "User" if role == "user" else "Assistant"
            lines.append(f"{prefix}: {text}")
        return "\n".join(lines)

    def update_summary(self):
        """
        아직 summary에 반영되지 않은 오래된 대화를 요약해서 summary에 병합
        recent window는 raw로 남기기 위해 제외
        """
        cutoff = max(0, len(self.full_history) - self.recent_window_size)
        if cutoff <= self.last_summarized_idx:
            return

        new_chunk = self.full_history[self.last_summarized_idx:cutoff]
        new_chunk_text = self._format_history_slice(new_chunk)

        summary_prompt = dedent(f"""
        You are a memory compression module for a conversational AI.

        Existing memory summary:
        {self.summary if self.summary else "(empty)"}

        New conversation chunk:
        {new_chunk_text}

        Update the memory summary by preserving only durable and useful facts, especially:
        - user's stable profile information
        - preferences
        - conditions and constraints
        - disambiguation between user facts and other people's facts
        - important corrections
        - important conversation context that may matter later

        Remove redundant chatter.
        Keep it concise but information-rich.
        Output only the updated summary.
        """).strip()

        summary_messages = [
            SystemMessage(content="You summarize conversation memory accurately."),
            HumanMessage(content=summary_prompt)
        ]
        response = self.llm.invoke(summary_messages)
        self.summary = response.content.strip()
        self.last_summarized_idx = cutoff

    def get_recent_messages(self):
        recent = self.full_history[-self.recent_window_size:]
        messages = []
        for role, text in recent:
            if role == "user":
                messages.append(HumanMessage(content=text))
            else:
                messages.append(AIMessage(content=text))
        return messages

    def build_context_messages(self, system_prompt):
        messages = [SystemMessage(content=system_prompt)]

        if self.summary.strip():
            memory_prompt = (
                "Long-term memory summary of previous conversation:\n"
                f"{self.summary}"
            )
            messages.append(SystemMessage(content=memory_prompt))

        messages.extend(self.get_recent_messages())
        return messages


# -----------------------------------
# 7. 메모리 객체 생성
# -----------------------------------
memory = SummaryMemory(llm=llm, recent_window_size=RECENT_WINDOW_SIZE)

BASE_SYSTEM_PROMPT = (
    "You are a careful assistant. "
    "Keep track of the user's profile accurately. "
    "Do not confuse the user's facts with facts about friends, family, or coworkers."
)

# -----------------------------------
# 8. 대화 시작
# -----------------------------------
initial_messages = [
    SystemMessage(content=BASE_SYSTEM_PROMPT),
    HumanMessage(content=PROFILE_TEXT)
]

resp = llm.invoke(initial_messages)

memory.add_user(PROFILE_TEXT)
memory.add_ai(resp.content)

print("=== INITIAL CONFIRMATION ===")
print(resp.content)
print()

# -----------------------------------
# 9. 테스트 함수
# -----------------------------------
def run_test(turn, memory_obj):
    print(f"\n{'='*80}")
    print(f"MEMORY TEST @ TURN {turn}")
    print(f"{'='*80}")

    total_score = 0
    max_score = len(QUESTION_BANK)

    for item in QUESTION_BANK:
        qname = item["name"]
        question = random.choice(PARAPHRASE_MAP[qname])

        local_messages = memory_obj.build_context_messages(BASE_SYSTEM_PROMPT)
        local_messages.append(HumanMessage(content=question))
        answer = llm.invoke(local_messages).content.strip()

        score = score_item(answer, item["targets"])
        total_score += score

        print(f"[{qname}]")
        print(f"Q: {question}")
        print(f"A: {answer}")
        print(f"Targets: {item['targets']}")
        print(f"Score: {score:.2f}")
        print("-" * 80)

    avg_score = total_score / max_score
    print(f"AVERAGE SCORE @ TURN {turn}: {avg_score:.3f}")

    if memory_obj.summary:
        print("\n[CURRENT MEMORY SUMMARY]")
        print(memory_obj.summary)
        print("-" * 80)

    return avg_score

# -----------------------------------
# 10. 간섭 대화 루프
# -----------------------------------
all_scores = []

for turn in range(1, TOTAL_TURNS + 1):
    noise1 = random.choice(LONG_NOISE)
    noise2 = random.choice(DISTRACTORS)
    noise3 = random.choice([
        "Please summarize the last message in one line.",
        "Is that fact about me or someone else?",
        "Acknowledge the last detail briefly.",
        "Extract the key entity from the previous message.",
    ])

    for text in [noise1, noise2, noise3]:
        context_messages = memory.build_context_messages(BASE_SYSTEM_PROMPT)
        context_messages.append(HumanMessage(content=text))

        resp = llm.invoke(context_messages)

        memory.add_user(text)
        memory.add_ai(resp.content)

    # 주기적으로 summary 갱신
    if turn % SUMMARY_UPDATE_EVERY == 0:
        memory.update_summary()

    # 주기적으로 테스트
    if turn % TEST_EVERY == 0:
        score = run_test(turn, memory)
        all_scores.append((turn, score))

# 마지막 summary 한 번 더 업데이트
memory.update_summary()

print("\n=== FINAL SUMMARY ===")
for turn, score in all_scores:
    print(f"Turn {turn:2d}: avg_score = {score:.3f}")

# -----------------------------------
# 11. 그래프
# -----------------------------------
turns = [t for t, s in all_scores]
scores = [s for t, s in all_scores]

plt.figure()
plt.plot(turns, scores, marker='o')
plt.xlabel("Turn")
plt.ylabel("Average Score")
plt.title("Memory Retention with Summary Memory")
plt.ylim(0, 1.05)
plt.grid()
plt.show()

