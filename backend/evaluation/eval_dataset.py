"""
Boston Pulse — RAG Evaluation Dataset
Test questions with expected datasets and answer keywords.
Used for measuring retrieval accuracy, answer faithfulness, and bias detection.
"""

EVAL_DATASET = [
    # =========================================================================
    # CRIME (6 questions)
    # =========================================================================
    {
        "id": "crime_01",
        "question": "What types of crimes were reported in district B2?",
        "expected_dataset": "crime",
        "expected_keywords": ["offense", "B2", "incident"],
        "category": "crime",
        "difficulty": "easy",
    },
    {
        "id": "crime_02",
        "question": "Were there any shooting incidents reported recently in Boston?",
        "expected_dataset": "crime",
        "expected_keywords": ["shooting", "incident"],
        "category": "crime",
        "difficulty": "easy",
    },
    {
        "id": "crime_03",
        "question": "What crimes happen most frequently at night after 10 PM?",
        "expected_dataset": "crime",
        "expected_keywords": ["hour", "offense", "night"],
        "category": "crime",
        "difficulty": "medium",
    },
    {
        "id": "crime_04",
        "question": "Is the area around Boylston Street safe to walk at night?",
        "expected_dataset": "crime",
        "expected_keywords": ["Boylston", "crime", "incident"],
        "category": "crime",
        "difficulty": "medium",
    },
    {
        "id": "crime_05",
        "question": "Which district has the most crime incidents on weekends?",
        "expected_dataset": "crime",
        "expected_keywords": ["district", "Saturday", "Sunday"],
        "category": "crime",
        "difficulty": "hard",
    },
    {
        "id": "crime_06",
        "question": "How does crime on Tremont Street compare between day and night?",
        "expected_dataset": "crime",
        "expected_keywords": ["Tremont", "hour", "offense"],
        "category": "crime",
        "difficulty": "hard",
    },

    # =========================================================================
    # 311 SERVICE REQUESTS (6 questions)
    # =========================================================================
    {
        "id": "311_01",
        "question": "What are the most common 311 service requests in Dorchester?",
        "expected_dataset": "service_311",
        "expected_keywords": ["Dorchester", "service", "request"],
        "category": "311",
        "difficulty": "easy",
    },
    {
        "id": "311_02",
        "question": "How long does it take for the city to resolve a pothole complaint?",
        "expected_dataset": "service_311",
        "expected_keywords": ["pothole", "status", "open"],
        "category": "311",
        "difficulty": "medium",
    },
    {
        "id": "311_03",
        "question": "Are 311 requests in Roxbury resolved on time?",
        "expected_dataset": "service_311",
        "expected_keywords": ["Roxbury", "on_time", "status"],
        "category": "311",
        "difficulty": "medium",
    },
    {
        "id": "311_04",
        "question": "What department handles street light outage complaints?",
        "expected_dataset": "service_311",
        "expected_keywords": ["street light", "department"],
        "category": "311",
        "difficulty": "easy",
    },
    {
        "id": "311_05",
        "question": "Which neighborhood has the most overdue 311 requests?",
        "expected_dataset": "service_311",
        "expected_keywords": ["neighborhood", "overdue", "status"],
        "category": "311",
        "difficulty": "hard",
    },
    {
        "id": "311_06",
        "question": "What types of service requests are filed most during winter months?",
        "expected_dataset": "service_311",
        "expected_keywords": ["service", "request", "topic"],
        "category": "311",
        "difficulty": "hard",
    },

    # =========================================================================
    # FOOD INSPECTIONS (6 questions)
    # =========================================================================
    {
        "id": "food_01",
        "question": "Which restaurants passed food inspections in Back Bay?",
        "expected_dataset": "food_inspections",
        "expected_keywords": ["restaurant", "result", "Pass"],
        "category": "food_inspections",
        "difficulty": "easy",
    },
    {
        "id": "food_02",
        "question": "Has the restaurant on 123 Main Street passed its health inspection?",
        "expected_dataset": "food_inspections",
        "expected_keywords": ["restaurant", "result", "address"],
        "category": "food_inspections",
        "difficulty": "easy",
    },
    {
        "id": "food_03",
        "question": "Which restaurants failed their most recent food inspection?",
        "expected_dataset": "food_inspections",
        "expected_keywords": ["restaurant", "Fail", "result"],
        "category": "food_inspections",
        "difficulty": "medium",
    },
    {
        "id": "food_04",
        "question": "Are there any restaurants with repeated inspection failures?",
        "expected_dataset": "food_inspections",
        "expected_keywords": ["restaurant", "Fail", "inspection"],
        "category": "food_inspections",
        "difficulty": "hard",
    },
    {
        "id": "food_05",
        "question": "What is the inspection result for Tatte Bakery?",
        "expected_dataset": "food_inspections",
        "expected_keywords": ["Tatte", "result"],
        "category": "food_inspections",
        "difficulty": "medium",
    },
    {
        "id": "food_06",
        "question": "How many food establishments were inspected last month?",
        "expected_dataset": "food_inspections",
        "expected_keywords": ["inspection", "restaurant", "date"],
        "category": "food_inspections",
        "difficulty": "medium",
    },

    # =========================================================================
    # BERDO - Building Emissions (6 questions)
    # =========================================================================
    {
        "id": "berdo_01",
        "question": "Which buildings in Boston have the highest greenhouse gas emissions?",
        "expected_dataset": "berdo",
        "expected_keywords": ["emissions", "property", "CO2"],
        "category": "berdo",
        "difficulty": "easy",
    },
    {
        "id": "berdo_02",
        "question": "What is the energy star score for buildings on Beacon Street?",
        "expected_dataset": "berdo",
        "expected_keywords": ["energy_star", "Beacon", "score"],
        "category": "berdo",
        "difficulty": "medium",
    },
    {
        "id": "berdo_03",
        "question": "How much energy does Boston University's campus use?",
        "expected_dataset": "berdo",
        "expected_keywords": ["energy", "kBtu", "property"],
        "category": "berdo",
        "difficulty": "medium",
    },
    {
        "id": "berdo_04",
        "question": "Which property type has the highest emissions per building?",
        "expected_dataset": "berdo",
        "expected_keywords": ["property_type", "emissions"],
        "category": "berdo",
        "difficulty": "hard",
    },
    {
        "id": "berdo_05",
        "question": "Are residential buildings more energy efficient than commercial ones?",
        "expected_dataset": "berdo",
        "expected_keywords": ["energy", "property_type", "residential"],
        "category": "berdo",
        "difficulty": "hard",
    },
    {
        "id": "berdo_06",
        "question": "What is the total energy consumption for buildings in zip code 02116?",
        "expected_dataset": "berdo",
        "expected_keywords": ["energy", "02116", "kBtu"],
        "category": "berdo",
        "difficulty": "medium",
    },

    # =========================================================================
    # CITYSCORE (6 questions)
    # =========================================================================
    {
        "id": "city_01",
        "question": "What is Boston's current CityScore?",
        "expected_dataset": "cityscore",
        "expected_keywords": ["score", "metric"],
        "category": "cityscore",
        "difficulty": "easy",
    },
    {
        "id": "city_02",
        "question": "Which city metrics are performing below target?",
        "expected_dataset": "cityscore",
        "expected_keywords": ["metric", "target", "score"],
        "category": "cityscore",
        "difficulty": "medium",
    },
    {
        "id": "city_03",
        "question": "How has the streetlight outage score changed over the past month?",
        "expected_dataset": "cityscore",
        "expected_keywords": ["streetlight", "score", "month"],
        "category": "cityscore",
        "difficulty": "medium",
    },
    {
        "id": "city_04",
        "question": "Is the 311 call center meeting its performance targets?",
        "expected_dataset": "cityscore",
        "expected_keywords": ["311", "score", "target"],
        "category": "cityscore",
        "difficulty": "medium",
    },
    {
        "id": "city_05",
        "question": "Which CityScore metric has the lowest weekly score?",
        "expected_dataset": "cityscore",
        "expected_keywords": ["metric", "week_score"],
        "category": "cityscore",
        "difficulty": "hard",
    },
    {
        "id": "city_06",
        "question": "How does Boston's overall city performance trend look this year?",
        "expected_dataset": "cityscore",
        "expected_keywords": ["score", "metric", "trend"],
        "category": "cityscore",
        "difficulty": "hard",
    },

    # =========================================================================
    # STREET SWEEPING (6 questions)
    # =========================================================================
    {
        "id": "sweep_01",
        "question": "When is street sweeping on Beacon Street?",
        "expected_dataset": "street_sweeping",
        "expected_keywords": ["Beacon", "schedule", "season"],
        "category": "street_sweeping",
        "difficulty": "easy",
    },
    {
        "id": "sweep_02",
        "question": "Which streets in District 5 have tow zone enforcement for sweeping?",
        "expected_dataset": "street_sweeping",
        "expected_keywords": ["District", "tow", "street"],
        "category": "street_sweeping",
        "difficulty": "medium",
    },
    {
        "id": "sweep_03",
        "question": "Is there street sweeping in my neighborhood during winter?",
        "expected_dataset": "street_sweeping",
        "expected_keywords": ["season", "sweeping", "schedule"],
        "category": "street_sweeping",
        "difficulty": "easy",
    },
    {
        "id": "sweep_04",
        "question": "What side of Commonwealth Avenue is swept on Mondays?",
        "expected_dataset": "street_sweeping",
        "expected_keywords": ["Commonwealth", "side", "schedule"],
        "category": "street_sweeping",
        "difficulty": "medium",
    },
    {
        "id": "sweep_05",
        "question": "Which streets are swept every week versus bi-weekly?",
        "expected_dataset": "street_sweeping",
        "expected_keywords": ["week", "frequency", "street"],
        "category": "street_sweeping",
        "difficulty": "hard",
    },
    {
        "id": "sweep_06",
        "question": "When does the street sweeping season start and end in Boston?",
        "expected_dataset": "street_sweeping",
        "expected_keywords": ["season_start", "season_end"],
        "category": "street_sweeping",
        "difficulty": "easy",
    },

    # =========================================================================
    # CROSS-DATASET (6 questions — tests if retriever pulls from multiple sources)
    # =========================================================================
    {
        "id": "cross_01",
        "question": "Is Dorchester a safe neighborhood with good city services?",
        "expected_dataset": ["crime", "service_311"],
        "expected_keywords": ["Dorchester", "crime", "service"],
        "category": "cross_dataset",
        "difficulty": "hard",
    },
    {
        "id": "cross_02",
        "question": "What is the overall quality of life in Back Bay?",
        "expected_dataset": ["crime", "service_311", "food_inspections"],
        "expected_keywords": ["Back Bay", "safety", "service"],
        "category": "cross_dataset",
        "difficulty": "hard",
    },
    {
        "id": "cross_03",
        "question": "I want to move to Boston. Which area has low crime and good restaurants?",
        "expected_dataset": ["crime", "food_inspections"],
        "expected_keywords": ["crime", "restaurant", "inspection"],
        "category": "cross_dataset",
        "difficulty": "hard",
    },
    {
        "id": "cross_04",
        "question": "Are there any environmental or safety concerns on Tremont Street?",
        "expected_dataset": ["crime", "berdo"],
        "expected_keywords": ["Tremont", "emissions", "crime"],
        "category": "cross_dataset",
        "difficulty": "hard",
    },
    {
        "id": "cross_05",
        "question": "How well is the city maintaining my neighborhood in Roxbury?",
        "expected_dataset": ["service_311", "cityscore", "street_sweeping"],
        "expected_keywords": ["Roxbury", "service", "sweeping", "score"],
        "category": "cross_dataset",
        "difficulty": "hard",
    },
    {
        "id": "cross_06",
        "question": "What should I know before renting an apartment in Jamaica Plain?",
        "expected_dataset": ["crime", "service_311", "food_inspections", "street_sweeping"],
        "expected_keywords": ["Jamaica Plain", "safety", "service"],
        "category": "cross_dataset",
        "difficulty": "hard",
    },
]


def get_eval_questions() -> list:
    """Return all evaluation questions."""
    return EVAL_DATASET


def get_questions_by_dataset(dataset: str) -> list:
    """Return evaluation questions for a specific dataset."""
    return [
        q for q in EVAL_DATASET
        if q["expected_dataset"] == dataset
        or (isinstance(q["expected_dataset"], list) and dataset in q["expected_dataset"])
    ]


def get_questions_by_difficulty(difficulty: str) -> list:
    """Return evaluation questions by difficulty level."""
    return [q for q in EVAL_DATASET if q["difficulty"] == difficulty]


def get_questions_by_category(category: str) -> list:
    """Return evaluation questions by category."""
    return [q for q in EVAL_DATASET if q["category"] == category]


# Quick summary
if __name__ == "__main__":
    total = len(EVAL_DATASET)
    by_cat = {}
    by_diff = {}
    for q in EVAL_DATASET:
        by_cat[q["category"]] = by_cat.get(q["category"], 0) + 1
        by_diff[q["difficulty"]] = by_diff.get(q["difficulty"], 0) + 1

    print(f"Total evaluation questions: {total}")
    print(f"\nBy category:")
    for cat, count in sorted(by_cat.items()):
        print(f"  {cat}: {count}")
    print(f"\nBy difficulty:")
    for diff, count in sorted(by_diff.items()):
        print(f"  {diff}: {count}")