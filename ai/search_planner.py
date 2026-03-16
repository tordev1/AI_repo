"""
Search-Based Study Planner
Compares A* Search vs Greedy Best-First Search for generating weekly study schedules.

State: partially filled weekly schedule (dict mapping day -> list of (subject, hours))
Goal: all subjects allocated within available study hours
Heuristic: deadline urgency + difficulty weighting
"""

import heapq
import copy
from dataclasses import dataclass, field


# ── Data Structures ───────────────────────────────────────────

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


@dataclass
class Subject:
    name: str
    hours_needed: float  # total hours needed this week
    days_to_deadline: int
    difficulty: float  # 1-10 scale

    @property
    def urgency_score(self):
        """Higher = more urgent. Combines deadline proximity and difficulty."""
        deadline_factor = max(0, 10 - self.days_to_deadline) / 10  # 0-1
        difficulty_factor = self.difficulty / 10  # 0-1
        return 0.6 * deadline_factor + 0.4 * difficulty_factor


@dataclass(order=True)
class ScheduleState:
    priority: float
    schedule: dict = field(compare=False)
    remaining: dict = field(compare=False)  # subject_name -> hours still to allocate
    cost: float = field(default=0.0, compare=False)  # g(n) — total cost so far
    path: list = field(default_factory=list, compare=False)  # sequence of actions taken

    def is_goal(self):
        return all(h <= 0.01 for h in self.remaining.values())

    def get_schedule_copy(self):
        return {day: list(slots) for day, slots in self.schedule.items()}


# ── Heuristic ─────────────────────────────────────────────────

def heuristic(state, subjects_map):
    """
    Estimates remaining cost to reach goal.
    Penalises subjects with closer deadlines and higher difficulty that still need hours.
    """
    h = 0.0
    for subj_name, hours_left in state.remaining.items():
        if hours_left > 0:
            subj = subjects_map[subj_name]
            # More urgent subjects contribute more to heuristic
            h += hours_left * (1 + subj.urgency_score)
    return h


# ── Generate Successor States ─────────────────────────────────

def get_successors(state, subjects_map, max_hours_per_day, day_hours):
    """
    Generate next states by allocating one block (1 hour) of a subject to a day.
    """
    successors = []
    for day in DAYS:
        current_day_hours = sum(h for _, h in state.schedule[day])
        available = day_hours.get(day, max_hours_per_day) - current_day_hours
        if available < 1:
            continue

        for subj_name, hours_left in state.remaining.items():
            if hours_left <= 0.01:
                continue

            subj = subjects_map[subj_name]
            block = min(1.0, hours_left, available)

            new_schedule = state.get_schedule_copy()
            new_schedule[day].append((subj_name, block))

            new_remaining = dict(state.remaining)
            new_remaining[subj_name] = round(hours_left - block, 2)

            action = f"Allocate {block}h of {subj_name} on {day}"
            new_path = state.path + [action]

            # Cost: slight penalty for splitting across days (encourages focused sessions)
            step_cost = block * (1 - 0.1 * subj.urgency_score)

            successors.append(ScheduleState(
                priority=0,  # will be set by search algorithm
                schedule=new_schedule,
                remaining=new_remaining,
                cost=state.cost + step_cost,
                path=new_path,
            ))
    return successors


# ── A* Search ─────────────────────────────────────────────────

def astar_search(subjects, max_hours_per_day=6, day_hours=None):
    """
    A* search: f(n) = g(n) + h(n)
    Returns (schedule, stats) or (None, stats) if no solution found.
    """
    if day_hours is None:
        day_hours = {d: max_hours_per_day for d in DAYS}

    subjects_map = {s.name: s for s in subjects}
    initial_remaining = {s.name: s.hours_needed for s in subjects}
    initial_schedule = {day: [] for day in DAYS}

    start = ScheduleState(
        priority=0,
        schedule=initial_schedule,
        remaining=initial_remaining,
        cost=0.0,
        path=[],
    )

    h = heuristic(start, subjects_map)
    start.priority = 0 + h

    frontier = [start]
    explored = 0
    max_explored = 5000  # safety limit

    while frontier and explored < max_explored:
        current = heapq.heappop(frontier)
        explored += 1

        if current.is_goal():
            stats = {"algorithm": "A*", "nodes_explored": explored}
            return current.schedule, stats

        for succ in get_successors(current, subjects_map, max_hours_per_day, day_hours):
            h = heuristic(succ, subjects_map)
            succ.priority = succ.cost + h
            heapq.heappush(frontier, succ)

    return None, {"algorithm": "A*", "nodes_explored": explored, "status": "limit_reached"}


# ── Greedy Best-First Search ──────────────────────────────────

def greedy_search(subjects, max_hours_per_day=6, day_hours=None):
    """
    Greedy Best-First: f(n) = h(n) only (ignores path cost)
    Returns (schedule, stats) or (None, stats) if no solution found.
    """
    if day_hours is None:
        day_hours = {d: max_hours_per_day for d in DAYS}

    subjects_map = {s.name: s for s in subjects}
    initial_remaining = {s.name: s.hours_needed for s in subjects}
    initial_schedule = {day: [] for day in DAYS}

    start = ScheduleState(
        priority=0,
        schedule=initial_schedule,
        remaining=initial_remaining,
        cost=0.0,
        path=[],
    )
    start.priority = heuristic(start, subjects_map)

    frontier = [start]
    explored = 0
    max_explored = 5000

    while frontier and explored < max_explored:
        current = heapq.heappop(frontier)
        explored += 1

        if current.is_goal():
            stats = {"algorithm": "Greedy Best-First", "nodes_explored": explored}
            return current.schedule, stats

        for succ in get_successors(current, subjects_map, max_hours_per_day, day_hours):
            h = heuristic(succ, subjects_map)
            succ.priority = h  # Greedy: only heuristic, no path cost
            heapq.heappush(frontier, succ)

    return None, {"algorithm": "Greedy Best-First", "nodes_explored": explored, "status": "limit_reached"}


# ── Convenience ───────────────────────────────────────────────

def create_schedule(subjects_data, available_hours_per_day=6, day_hours=None):
    """
    High-level function: runs both algorithms and returns comparison.
    subjects_data: list of dicts with keys name, hours_needed, days_to_deadline, difficulty
    Returns dict with astar_schedule, greedy_schedule, comparison stats.
    """
    subjects = [Subject(**s) for s in subjects_data]

    astar_schedule, astar_stats = astar_search(subjects, available_hours_per_day, day_hours)
    greedy_schedule, greedy_stats = greedy_search(subjects, available_hours_per_day, day_hours)

    return {
        "astar": {"schedule": astar_schedule, "stats": astar_stats},
        "greedy": {"schedule": greedy_schedule, "stats": greedy_stats},
        "subjects": subjects_data,
    }


def format_schedule(schedule):
    """Format a schedule dict into a readable string."""
    if schedule is None:
        return "No valid schedule found within search limits."
    lines = []
    for day in DAYS:
        slots = schedule[day]
        if slots:
            merged = {}
            for subj, hours in slots:
                merged[subj] = merged.get(subj, 0) + hours
            parts = [f"{subj} ({h:.0f}h)" for subj, h in merged.items()]
            lines.append(f"  {day}: {', '.join(parts)}")
        else:
            lines.append(f"  {day}: Free")
    return "\n".join(lines)


# ── Main (demo) ───────────────────────────────────────────────

def main():
    subjects_data = [
        {"name": "AI Coursework", "hours_needed": 6, "days_to_deadline": 5, "difficulty": 8},
        {"name": "Databases", "hours_needed": 4, "days_to_deadline": 10, "difficulty": 6},
        {"name": "Networking", "hours_needed": 3, "days_to_deadline": 14, "difficulty": 5},
    ]

    result = create_schedule(subjects_data, available_hours_per_day=4)

    print("=== A* Search Schedule ===")
    print(format_schedule(result["astar"]["schedule"]))
    print(f"Nodes explored: {result['astar']['stats']['nodes_explored']}")

    print("\n=== Greedy Best-First Schedule ===")
    print(format_schedule(result["greedy"]["schedule"]))
    print(f"Nodes explored: {result['greedy']['stats']['nodes_explored']}")


if __name__ == "__main__":
    main()
