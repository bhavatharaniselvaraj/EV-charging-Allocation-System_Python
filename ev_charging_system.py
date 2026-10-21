"""
EV Charging Station Allocation and Conflict Management System
=============================================================
Terminal application, Python 3.8+, standard library only.

Main data structure : Interval Tree (one tree per charging station)
Interval convention : half-open  [start, end)
                      so [10:00, 11:00) and [11:00, 12:00) do NOT overlap.
Times               : stored internally as minutes after midnight (int).

Run:      python ev_charging_system.py            (empty system)
          python ev_charging_system.py --demo     (with sample data)
Admin key: "admin123" by default, or set the environment variable EV_ADMIN_KEY.

Booking rules
  * Every EV ID must be unique: an ID that already has a reservation or a
    waiting-list request cannot be used for another vehicle.
  * The service time is shown automatically when a reservation is booked.
  * If the preferred station is busy the EV is offered the other free
    stations, but is moved ONLY with the user's explicit permission.
    If the user declines (or no other station is free) the request goes to
    the WAITING LIST.

File layout
  1. Time helpers
  2. IntervalNode / IntervalTree          <- the core data structure
  3. Station, Reservation, WaitingEntry   <- plain data holders
  4. ChargingSystem                       <- stations, reservations, conflicts,
                                             alternatives, waiting list
  5. Admin authentication + Admin menu
  6. User menu
  7. main()
"""

import hmac
import os
import re
import sys
from getpass import getpass

ADMIN_SECRET_KEY = os.environ.get("EV_ADMIN_KEY", "admin123")
MINUTES_PER_DAY = 24 * 60
SLOT_STEP = 15          # minutes between candidate slots when searching alternatives


# ======================================================================
# 1. TIME HELPERS
# ======================================================================
def parse_time(text):
    """'09:30' -> 570 (minutes after midnight). Raises ValueError if invalid."""
    match = re.fullmatch(r"(\d{1,2}):(\d{2})", str(text).strip())
    if not match:
        raise ValueError("Time must be in HH:MM format, e.g. 09:30.")
    hours, minutes = int(match.group(1)), int(match.group(2))
    if minutes > 59 or hours > 24 or (hours == 24 and minutes != 0):
        raise ValueError("Time out of range (00:00 - 24:00).")
    return hours * 60 + minutes


def format_time(minutes):
    """570 -> '09:30'."""
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def validate_interval(start, end):
    """Start must be strictly earlier than end."""
    if start >= end:
        raise ValueError("Start time must be earlier than end time.")


def service_time(start, end):
    """Service Time = End Time - Start Time (minutes)."""
    validate_interval(start, end)
    return end - start


def waiting_time(arrival, charging_start):
    """Waiting Time = Actual Charging Start Time - Arrival Time (minutes).
    Calculated separately from the Interval Tree's data."""
    if charging_start < arrival:
        raise ValueError("Charging cannot start before the EV arrives.")
    return charging_start - arrival


# ======================================================================
# 2. INTERVAL TREE
# ======================================================================
def overlaps(s1, e1, s2, e2):
    """Half-open overlap test: [s1,e1) and [s2,e2) overlap iff each one
    starts before the other ends.  Touching intervals (e1 == s2) do not overlap."""
    return s1 < e2 and s2 < e1


class IntervalNode:
    """One reservation stored in the tree."""

    def __init__(self, reservation_id, ev_id, start, end):
        self.reservation_id = reservation_id
        self.ev_id = ev_id
        self.start = start
        self.end = end
        self.max_end_time = end      # largest 'end' in this node's whole subtree
        self.left = None
        self.right = None

    def key(self):
        # Ordered by start, then end, then reservation ID. Including the ID
        # means two reservations with identical times are still distinct keys.
        return (self.start, self.end, self.reservation_id)


class IntervalTree:
    """A binary search tree ordered by interval start time, where every node
    also keeps max_end_time for its subtree.  That value lets the overlap
    search skip whole subtrees that cannot contain a conflict.

    (Kept as a plain, unbalanced BST for readability. Operations are O(h).)"""

    def __init__(self):
        self.root = None
        self.size = 0

    # ---------- maxEndTime maintenance ----------
    @staticmethod
    def _update_max(node):
        """Recompute max_end_time from the node itself and its two children."""
        best = node.end
        if node.left is not None and node.left.max_end_time > best:
            best = node.left.max_end_time
        if node.right is not None and node.right.max_end_time > best:
            best = node.right.max_end_time
        node.max_end_time = best

    # ---------- insert ----------
    def insert(self, reservation_id, ev_id, start, end):
        """Insert an interval (the caller checks for overlap first)."""
        self.root = self._insert(self.root, IntervalNode(reservation_id, ev_id, start, end))
        self.size += 1

    def _insert(self, node, new_node):
        if node is None:
            return new_node
        if new_node.key() < node.key():
            node.left = self._insert(node.left, new_node)
        else:
            node.right = self._insert(node.right, new_node)
        self._update_max(node)           # fix maxEndTime on the way back up
        return node

    # ---------- delete ----------
    def delete(self, reservation_id, start, end):
        """Delete the node with this exact (start, end, id). Returns True if found."""
        found = [False]
        self.root = self._delete(self.root, (start, end, reservation_id), found)
        if found[0]:
            self.size -= 1
        return found[0]

    def _delete(self, node, key, found):
        if node is None:
            return None
        if key < node.key():
            node.left = self._delete(node.left, key, found)
        elif key > node.key():
            node.right = self._delete(node.right, key, found)
        else:
            found[0] = True
            if node.left is None:
                return node.right
            if node.right is None:
                return node.left
            # Two children: copy the in-order successor's data here, then
            # delete the successor from the right subtree.
            successor = node.right
            while successor.left is not None:
                successor = successor.left
            succ_key = successor.key()
            node.reservation_id = successor.reservation_id
            node.ev_id = successor.ev_id
            node.start = successor.start
            node.end = successor.end
            node.right = self._delete(node.right, succ_key, [False])
        self._update_max(node)           # fix maxEndTime on the way back up
        return node

    # ---------- overlap search ----------
    def find_overlap(self, start, end):
        """Return ONE node overlapping [start, end), or None (stops early)."""
        return self._first_overlap(self.root, start, end)

    def _first_overlap(self, node, start, end):
        if node is None:
            return None
        # Left subtree can only overlap if something in it ends after 'start'.
        if node.left is not None and node.left.max_end_time > start:
            found = self._first_overlap(node.left, start, end)
            if found is not None:
                return found
        if overlaps(node.start, node.end, start, end):
            return node
        # Right subtree starts at >= node.start; useless if that is already >= end.
        if node.right is not None and node.start < end and node.right.max_end_time > start:
            return self._first_overlap(node.right, start, end)
        return None

    def find_all_overlaps(self, start, end):
        """Return a list of ALL nodes overlapping [start, end)."""
        result = []
        self._collect_overlaps(self.root, start, end, result)
        return result

    def _collect_overlaps(self, node, start, end, result):
        if node is None:
            return
        if node.left is not None and node.left.max_end_time > start:
            self._collect_overlaps(node.left, start, end, result)
        if overlaps(node.start, node.end, start, end):
            result.append(node)
        if node.right is not None and node.start < end and node.right.max_end_time > start:
            self._collect_overlaps(node.right, start, end, result)

    def has_overlap(self, start, end):
        return self.find_overlap(start, end) is not None

    # ---------- traversal and lookups ----------
    def inorder(self):
        """All nodes sorted by start time (in-order traversal)."""
        result = []
        self._inorder(self.root, result)
        return result

    def _inorder(self, node, result):
        if node is None:
            return
        self._inorder(node.left, result)
        result.append(node)
        self._inorder(node.right, result)

    def find_by_id(self, reservation_id):
        """Traversal search (the ID is not the tree's key, so we scan)."""
        for node in self.inorder():
            if node.reservation_id == reservation_id:
                return node
        return None

    def find_by_ev(self, ev_id):
        return [n for n in self.inorder() if n.ev_id == ev_id]

    # ---------- self-check used by the tests ----------
    def is_valid(self):
        """Check BST order, correct maxEndTime everywhere, and the size counter."""
        count = [0]

        def check(node, low, high):
            if node is None:
                return True
            count[0] += 1
            if low is not None and node.key() < low:
                return False
            if high is not None and node.key() >= high:
                return False
            expected = node.end
            for child in (node.left, node.right):
                if child is not None:
                    expected = max(expected, child.max_end_time)
            if node.max_end_time != expected:
                return False
            return check(node.left, low, node.key()) and check(node.right, node.key(), high)

        return check(self.root, None, None) and count[0] == self.size


# ======================================================================
# 3. SIMPLE DATA HOLDERS
# ======================================================================
class Station:
    def __init__(self, station_id, name, location, capacity_kw):
        self.station_id = station_id
        self.name = name
        self.location = location
        self.capacity_kw = capacity_kw
        self.tree = IntervalTree()       # every station owns its own tree


class Reservation:
    """Reservation record. The tree stores the interval; this record also
    keeps the arrival time, which is used only for waiting-time calculations."""

    def __init__(self, reservation_id, ev_id, station_id, start, end, arrival):
        self.reservation_id = reservation_id
        self.ev_id = ev_id
        self.station_id = station_id
        self.start = start
        self.end = end
        self.arrival = arrival

    @property
    def duration(self):
        return self.end - self.start


class WaitingEntry:
    """A request that could not be served. stations=None means 'any station'."""

    def __init__(self, ev_id, stations, start, end, arrival):
        self.ev_id = ev_id
        self.stations = stations
        self.start = start
        self.end = end
        self.arrival = arrival


class ConflictError(ValueError):
    """Raised when a requested interval overlaps an existing reservation."""


# ======================================================================
# 4. CHARGING SYSTEM (all business logic, no input()/print())
# ======================================================================
class ChargingSystem:
    def __init__(self):
        self.stations = {}        # station_id -> Station
        self.reservations = {}    # reservation_id -> Reservation (supporting index)
        self.waiting = []         # simple list of WaitingEntry (first come, first served)
        self._counter = 0

    # ---------- unique EV IDs ----------
    def ev_id_in_use(self, ev_id):
        """True if this EV ID already has a reservation or a waiting-list request."""
        ev_id = ev_id.strip().upper()
        return (any(r.ev_id == ev_id for r in self.reservations.values())
                or any(e.ev_id == ev_id for e in self.waiting))

    def require_unique_ev(self, ev_id):
        if not ev_id.strip():
            raise ValueError("EV ID cannot be empty.")
        if self.ev_id_in_use(ev_id):
            raise ValueError(f"EV ID '{ev_id.strip().upper()}' is already registered to another "
                             "vehicle. Please use a different EV ID.")

    # ---------- station management ----------
    def add_station(self, station_id, name, location, capacity_kw):
        station_id = station_id.strip().upper()
        if not station_id:
            raise ValueError("Station ID cannot be empty.")
        if station_id in self.stations:
            raise ValueError(f"Station ID '{station_id}' already exists.")
        if not name.strip() or not location.strip():
            raise ValueError("Station name and location cannot be empty.")
        if capacity_kw <= 0:
            raise ValueError("Charging capacity must be a positive number.")
        self.stations[station_id] = Station(station_id, name.strip(), location.strip(), capacity_kw)
        return self.stations[station_id]

    def get_station(self, station_id):
        station = self.stations.get(station_id.strip().upper())
        if station is None:
            raise ValueError(f"Station '{station_id.strip().upper()}' does not exist.")
        return station

    def station_load(self, station_id):
        """(number of reservations, booked minutes, congestion label)."""
        nodes = self.get_station(station_id).tree.inorder()
        booked = sum(n.end - n.start for n in nodes)
        share = booked / MINUTES_PER_DAY
        label = "Low" if share < 0.25 else "Moderate" if share < 0.60 else "High"
        return len(nodes), booked, label

    def delete_station(self, station_id, resolution=None):
        """Delete a station. If it still has reservations, 'resolution' must be
        'relocate' (move them elsewhere, otherwise to the waiting list) or
        'cancel' (cancel them). Nothing is silently dropped.
        Returns a summary dict."""
        station = self.get_station(station_id)
        sid = station.station_id
        summary = {"moved": [], "waitlisted": [], "cancelled": []}

        if station.tree.size > 0:
            if resolution not in ("relocate", "cancel"):
                raise ValueError(
                    f"Station {sid} has {station.tree.size} active reservation(s). "
                    "Resolve them (relocate or cancel) before deleting.")
            for node in station.tree.inorder():
                res = self.reservations.pop(node.reservation_id)
                if resolution == "relocate":
                    new_sid = next((s for s in self.free_stations(res.start, res.end, exclude=sid)), None)
                    if new_sid is not None:
                        self._place(res.reservation_id, res.ev_id, new_sid, res.start, res.end, res.arrival)
                        summary["moved"].append((res.reservation_id, new_sid))
                    else:
                        self.waiting.append(WaitingEntry(res.ev_id, None, res.start, res.end, res.arrival))
                        summary["waitlisted"].append(res.reservation_id)
                else:
                    summary["cancelled"].append(res.reservation_id)

        del self.stations[sid]
        # Waiting entries must not point to a station that no longer exists.
        for entry in self.waiting:
            if entry.stations is not None:
                entry.stations = [s for s in entry.stations if s != sid]
                if not entry.stations:
                    entry.stations = None     # their only choice is gone -> accept any station
        return summary

    # ---------- conflict detection ----------
    def is_free(self, station_id, start, end, ignore_rid=None):
        """True if [start,end) does not overlap anything in the station's tree.
        'ignore_rid' lets an update ignore the reservation being changed."""
        tree = self.get_station(station_id).tree
        if ignore_rid is None:
            return not tree.has_overlap(start, end)
        return all(n.reservation_id == ignore_rid for n in tree.find_all_overlaps(start, end))

    def free_stations(self, start, end, exclude=None, ignore_rid=None):
        """IDs of all stations (except 'exclude') that can take the interval."""
        return [sid for sid in sorted(self.stations)
                if sid != exclude and self.is_free(sid, start, end, ignore_rid)]

    def suggest_time_slots(self, station_ids, start, end, min_start=0, ignore_rid=None, limit=5):
        """Look for free slots of the SAME length near the requested time, in steps
        of SLOT_STEP minutes (earlier and later). Returns [(station_id, start, end)]."""
        length = end - start
        found = []
        for k in range(1, MINUTES_PER_DAY // SLOT_STEP + 1):
            for cand in (start + k * SLOT_STEP, start - k * SLOT_STEP):
                if cand < min_start or cand < 0 or cand + length > MINUTES_PER_DAY:
                    continue
                for sid in station_ids:
                    if sid in self.stations and self.is_free(sid, cand, cand + length, ignore_rid):
                        found.append((sid, cand, cand + length))
                        break                       # one station per time is enough
                if len(found) >= limit:
                    return found
        return found

    # ---------- reservations ----------
    def _new_id(self):
        self._counter += 1
        return f"R{self._counter:03d}"

    def _place(self, reservation_id, ev_id, station_id, start, end, arrival):
        """Insert into the tree AND the record index together (keeps them consistent)."""
        self.stations[station_id].tree.insert(reservation_id, ev_id, start, end)
        res = Reservation(reservation_id, ev_id, station_id, start, end, arrival)
        self.reservations[reservation_id] = res
        return res

    def book(self, ev_id, station_id, start, end, arrival):
        """Book a slot. Raises ValueError (invalid) or ConflictError (overlap)."""
        ev_id = ev_id.strip().upper()
        self.require_unique_ev(ev_id)
        validate_interval(start, end)
        if arrival > start:
            raise ValueError("Arrival time cannot be later than the charging start time.")
        station = self.get_station(station_id)
        if station.tree.has_overlap(start, end):
            raise ConflictError(f"{format_time(start)}-{format_time(end)} is already booked at {station.station_id}.")
        return self._place(self._new_id(), ev_id, station.station_id, start, end, arrival)

    def cancel(self, reservation_id):
        """Cancel a reservation, then try to give the freed slot to a waiting EV.
        Returns (cancelled_reservation, [newly_allocated_reservations])."""
        res = self.reservations.get(reservation_id.strip().upper())
        if res is None:
            raise ValueError(f"Reservation '{reservation_id.strip().upper()}' does not exist.")
        self.stations[res.station_id].tree.delete(res.reservation_id, res.start, res.end)
        del self.reservations[res.reservation_id]
        return res, self.process_waiting(res.station_id)

    def update(self, reservation_id, new_station_id, new_start, new_end):
        """Change station and/or interval. The old interval is removed first; if the
        new one conflicts, the original is put back exactly as it was and
        ConflictError is raised. Returns (reservation, [waiting EVs allocated])."""
        res = self.reservations.get(reservation_id.strip().upper())
        if res is None:
            raise ValueError(f"Reservation '{reservation_id.strip().upper()}' does not exist.")
        validate_interval(new_start, new_end)
        new_station = self.get_station(new_station_id)
        old_station = self.stations[res.station_id]

        old_station.tree.delete(res.reservation_id, res.start, res.end)       # remove old
        if new_station.tree.has_overlap(new_start, new_end):                  # check new
            old_station.tree.insert(res.reservation_id, res.ev_id, res.start, res.end)  # restore
            raise ConflictError("The new interval conflicts with another reservation. "
                                "Your original reservation is unchanged.")
        new_station.tree.insert(res.reservation_id, res.ev_id, new_start, new_end)

        old_sid = res.station_id
        res.station_id, res.start, res.end = new_station.station_id, new_start, new_end
        if res.arrival > new_start:
            res.arrival = new_start      # keep waiting-time data valid
        return res, self.process_waiting(old_sid)       # the old slot is now free

    # ---------- searching ----------
    def find_by_id(self, reservation_id):
        """Locate a reservation by walking each station's tree."""
        reservation_id = reservation_id.strip().upper()
        for station in self.stations.values():
            node = station.tree.find_by_id(reservation_id)
            if node is not None:
                return self.reservations[reservation_id]
        return None

    def find_by_ev(self, ev_id):
        ev_id = ev_id.strip().upper()
        found = []
        for sid in sorted(self.stations):
            for node in self.stations[sid].tree.find_by_ev(ev_id):
                found.append(self.reservations[node.reservation_id])
        return found

    def all_reservations(self, station_id=None):
        """Reservations in start-time order per station, via in-order traversal."""
        sids = [self.get_station(station_id).station_id] if station_id else sorted(self.stations)
        return [self.reservations[n.reservation_id] for sid in sids
                for n in self.stations[sid].tree.inorder()]

    # ---------- waiting list ----------
    def add_waiting(self, ev_id, stations, start, end, arrival):
        ev_id = ev_id.strip().upper()
        self.require_unique_ev(ev_id)
        validate_interval(start, end)
        entry = WaitingEntry(ev_id, stations, start, end, arrival)
        self.waiting.append(entry)
        return entry

    def remove_waiting(self, ev_id, start, end):
        """Take an EV's request off the waiting list. Returns True if it was there."""
        ev_id = ev_id.strip().upper()
        for entry in self.waiting:
            if entry.ev_id == ev_id and entry.start == start and entry.end == end:
                self.waiting.remove(entry)
                return True
        return False

    def process_waiting(self, freed_station_id=None):
        """Try to allocate waiting EVs (oldest first). An entry is removed from the
        list only AFTER its reservation has been created. Returns new reservations."""
        allocated = []
        for entry in list(self.waiting):
            wanted = entry.stations if entry.stations is not None else sorted(self.stations)
            if freed_station_id is not None:
                if freed_station_id not in wanted:
                    continue
                wanted = [freed_station_id]
            for sid in wanted:
                if sid in self.stations and self.is_free(sid, entry.start, entry.end):
                    res = self._place(self._new_id(), entry.ev_id, sid, entry.start, entry.end, entry.arrival)
                    self.waiting.remove(entry)
                    allocated.append(res)
                    break
        return allocated

    # ---------- consistency check (used by tests) ----------
    def verify_consistency(self):
        seen = set()
        for sid, station in self.stations.items():
            if not station.tree.is_valid():
                return False
            for node in station.tree.inorder():
                res = self.reservations.get(node.reservation_id)
                if (res is None or res.station_id != sid or res.start != node.start
                        or res.end != node.end or res.ev_id != node.ev_id):
                    return False
                seen.add(node.reservation_id)
            # no two intervals in one tree may overlap
            nodes = station.tree.inorder()
            for a, b in zip(nodes, nodes[1:]):
                if a.end > b.start:
                    return False
        return seen == set(self.reservations)


# ======================================================================
# Terminal input/output helpers
# ======================================================================
def ask(prompt):
    return input(prompt).strip()


def ask_time(prompt):
    while True:
        try:
            return parse_time(ask(prompt))
        except ValueError as err:
            print(f"  ! {err}")


def ask_positive_int(prompt):
    while True:
        text = ask(prompt)
        if text.isdigit() and int(text) > 0:
            return int(text)
        print("  ! Please enter a positive whole number.")


def ask_yes_no(prompt):
    while True:
        answer = ask(prompt + " (y/n): ").lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("  ! Please answer y or n.")


def ask_choice(prompt, count):
    """Pick 1..count, or 0 to skip. Returns the number."""
    while True:
        text = ask(prompt)
        if text.isdigit() and 0 <= int(text) <= count:
            return int(text)
        print(f"  ! Enter a number from 0 to {count}.")


def print_reservations(rows, show_ev=True):
    if not rows:
        print("  (no reservations)")
        return
    header = f"  {'Res ID':<8}{'EV ID':<10}{'Station':<9}{'Start':<7}{'End':<7}{'Service(min)':>13}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        ev = r.ev_id if show_ev else "-"
        print(f"  {r.reservation_id:<8}{ev:<10}{r.station_id:<9}{format_time(r.start):<7}"
              f"{format_time(r.end):<7}{r.duration:>13}")


def print_stations(system):
    if not system.stations:
        print("  (no charging stations registered)")
        return
    print(f"  {'ID':<6}{'Name':<20}{'Location':<20}{'kW':>5}{'Reservations':>14}  {'Congestion'}")
    print("  " + "-" * 78)
    for sid in sorted(system.stations):
        st = system.stations[sid]
        count, booked, label = system.station_load(sid)
        print(f"  {st.station_id:<6}{st.name:<20}{st.location:<20}{st.capacity_kw:>5}"
              f"{count:>14}  {label} ({booked} min booked)")


def print_confirmation(res):
    print("\n  *** RESERVATION CONFIRMED ***")
    print(f"  Reservation ID : {res.reservation_id}")
    print(f"  EV ID          : {res.ev_id}")
    print(f"  Station ID     : {res.station_id}")
    print(f"  Time           : {format_time(res.start)} - {format_time(res.end)}")
    print(f"  Service time   : {res.duration} minutes")


def print_waiting_list(system, show_ev=True):
    if not system.waiting:
        print("  (waiting list is empty)")
        return
    print(f"  {'#':<4}{'EV ID':<10}{'Stations':<14}{'Start':<7}{'End':<7}{'Arrival':<8}")
    print("  " + "-" * 50)
    for i, e in enumerate(system.waiting, 1):
        stations = "ANY" if e.stations is None else ",".join(e.stations)
        print(f"  {i:<4}{(e.ev_id if show_ev else '-'):<10}{stations:<14}"
              f"{format_time(e.start):<7}{format_time(e.end):<7}{format_time(e.arrival):<8}")


def report_allocated(allocated):
    for r in allocated:
        print(f"  >> Waiting list: EV {r.ev_id} was given the freed slot at {r.station_id} "
              f"({format_time(r.start)}-{format_time(r.end)}), reservation {r.reservation_id}, "
              f"service time {r.duration} minutes.")


# ======================================================================
# 5. ADMIN: authentication + menu
# ======================================================================
class AdminSession:
    """Admin access exists only while 'authenticated' is True."""

    def __init__(self):
        self.authenticated = False

    def login(self, key):
        # compare_digest avoids leaking information through timing
        if hmac.compare_digest(key.encode(), ADMIN_SECRET_KEY.encode()):
            self.authenticated = True
        return self.authenticated

    def logout(self):
        self.authenticated = False

    def require(self):
        if not self.authenticated:
            raise PermissionError("Admin authentication required.")


def read_secret(prompt):
    try:
        return getpass(prompt)
    except Exception:
        return input(prompt)


def admin_add_station(system, session):
    session.require()
    print("\n-- Add Charging Station --")
    try:
        sid = ask("Station ID: ")
        name = ask("Station name: ")
        location = ask("Location: ")
        capacity = ask_positive_int("Charging capacity (kW): ")
        st = system.add_station(sid, name, location, capacity)
        print(f"  Station {st.station_id} added.")
    except ValueError as err:
        print(f"  ! {err}")


def admin_view_stations(system, session):
    session.require()
    print("\n-- Charging Stations --")
    print_stations(system)


def admin_delete_station(system, session):
    session.require()
    print("\n-- Delete Charging Station --")
    if not system.stations:
        print("  (no charging stations registered)")
        return
    try:
        station = system.get_station(ask("Station ID to delete: "))
        resolution = None
        if station.tree.size > 0:
            print(f"  WARNING: {station.station_id} has {station.tree.size} active reservation(s).")
            print("  1. Relocate them to other stations (unplaceable ones go to the waiting list)")
            print("  2. Cancel all of them")
            print("  3. Abort")
            pick = ask_choice("Choose: ", 3)
            if pick in (0, 3):
                print("  Deletion aborted.")
                return
            resolution = "relocate" if pick == 1 else "cancel"
        if not ask_yes_no(f"Really delete station {station.station_id}?"):
            print("  Deletion aborted.")
            return
        summary = system.delete_station(station.station_id, resolution)
        print(f"  Station {station.station_id} deleted.")
        for rid, new_sid in summary["moved"]:
            print(f"   - {rid} moved to {new_sid}")
        for rid in summary["waitlisted"]:
            print(f"   - {rid} could not be moved; its EV is now on the waiting list")
        for rid in summary["cancelled"]:
            print(f"   - {rid} cancelled")
    except ValueError as err:
        print(f"  ! {err}")


def admin_view_reservations(system, session):
    session.require()
    print("\n-- View Reservations --")
    if not system.stations:
        print("  (no charging stations registered)")
        return
    print("  1. All stations   2. One station")
    pick = ask_choice("Choose: ", 2)
    try:
        if pick == 2:
            rows = system.all_reservations(ask("Station ID: "))
        else:
            rows = system.all_reservations()
    except ValueError as err:
        print(f"  ! {err}")
        return
    print_reservations(rows)


def admin_search(system, session):
    session.require()
    print("\n-- Search Reservation --")
    print("  1. By Reservation ID   2. By EV ID")
    pick = ask_choice("Choose: ", 2)
    if pick == 1:
        res = system.find_by_id(ask("Reservation ID: "))
        rows = [res] if res else []
    elif pick == 2:
        rows = system.find_by_ev(ask("EV ID: "))
    else:
        return
    if not rows:
        print("  No matching reservation exists.")
        return
    print_reservations(rows)
    for r in rows:
        st = system.stations[r.station_id]
        print(f"  Station {st.station_id}: {st.name}, {st.location}, {st.capacity_kw} kW")


def admin_cancel(system, session):
    session.require()
    print("\n-- Cancel Reservation --")
    try:
        res, allocated = system.cancel(ask("Reservation ID: "))
        print(f"  Reservation {res.reservation_id} (EV {res.ev_id}, {res.station_id} "
              f"{format_time(res.start)}-{format_time(res.end)}) cancelled; slot released.")
        report_allocated(allocated)
    except ValueError as err:
        print(f"  ! {err}")


def admin_update(system, session):
    session.require()
    print("\n-- Update Reservation --")
    try:
        res = system.find_by_id(ask("Reservation ID: "))
        if res is None:
            print("  ! Reservation does not exist.")
            return
        print(f"  Current: {res.station_id} {format_time(res.start)}-{format_time(res.end)}")
        sid = ask("New station ID (blank = keep): ") or res.station_id
        start = ask_time("New start (HH:MM): ")
        end = ask_time("New end (HH:MM): ")
        res, allocated = system.update(res.reservation_id, sid, start, end)
        print(f"  Updated: {res.reservation_id} now at {res.station_id} "
              f"{format_time(res.start)}-{format_time(res.end)}")
        report_allocated(allocated)
    except ConflictError as err:
        print(f"  ! Conflict: {err}")
    except ValueError as err:
        print(f"  ! {err}")


def admin_waiting(system, session):
    session.require()
    print("\n-- Waiting List --")
    print_waiting_list(system)
    if system.waiting:
        allocated = system.process_waiting()
        if allocated:
            report_allocated(allocated)
        else:
            print("  No waiting EV can be allocated right now.")


def admin_menu(system):
    session = AdminSession()
    key = read_secret("Enter admin secret key: ")
    if not session.login(key):
        print("  ! Incorrect secret key. Access denied.")
        return
    print("  Admin access granted.")
    actions = {
        "1": admin_add_station, "2": admin_view_stations, "3": admin_delete_station,
        "4": admin_view_reservations, "5": admin_search, "6": admin_cancel,
        "7": admin_update, "8": admin_waiting,
    }
    while session.authenticated:
        print("\n===== ADMIN MENU =====")
        print("1. Add Charging Station\n2. View All Charging Stations\n3. Delete Charging Station")
        print("4. View All Reservations\n5. Search Reservation\n6. Cancel Reservation")
        print("7. Update Reservation\n8. View Waiting List\n9. Logout")
        choice = ask("Choose an option: ")
        if choice == "9":
            session.logout()
            print("  Logged out.")
        elif choice in actions:
            actions[choice](system, session)
        else:
            print("  ! Invalid option.")


# ======================================================================
# 6. USER MENU
# ======================================================================
def ask_request(system, need_station=True, new_ev=True):
    """Collect and validate EV ID, station, arrival, start and end.
    new_ev=True: the EV ID must be unique (not used by another vehicle); the user is
    asked again until a free ID is entered (blank = cancel).
    Returns (ev_id, station_id, arrival, start, end) or None on invalid input."""
    try:
        ev_id = ""
        if new_ev:
            while True:
                ev_id = ask("EV ID: ").upper()
                if not ev_id:
                    raise ValueError("EV ID cannot be empty.")
                if not system.ev_id_in_use(ev_id):
                    break
                print(f"  ! EV ID '{ev_id}' is already taken by another vehicle. "
                      "Please enter a different EV ID.")
        station_id = ""
        if need_station:
            station_id = system.get_station(ask("Preferred Station ID: ")).station_id
        arrival = ask_time("Arrival time (HH:MM): ")
        start = ask_time("Charging start (HH:MM): ")
        end = ask_time("Charging end (HH:MM): ")
        validate_interval(start, end)
        if arrival > start:
            raise ValueError("Arrival time cannot be later than the charging start time.")
        return ev_id, station_id, arrival, start, end
    except ValueError as err:
        print(f"  ! {err}")
        return None


def offer_alternative_stations(system, ev_id, preferred, arrival, start, end):
    """Offer other free stations. The EV is moved ONLY if the user gives permission.
    Returns True if a reservation was made at an alternative station."""
    options = system.free_stations(start, end, exclude=preferred)
    if not options:
        print("  No other station is free for that interval.")
        return False
    print("  These other stations are free for your interval:")
    for i, sid in enumerate(options, 1):
        st = system.stations[sid]
        print(f"   {i}. {sid} - {st.name}, {st.location}")
    if not ask_yes_no(f"  Do you give permission to redirect your booking away from {preferred}?"):
        print("  Redirect declined. Your booking will not be moved.")
        return False
    pick = ask_choice("Choose a station (0 to cancel the redirect): ", len(options))
    if pick == 0:
        print("  Redirect cancelled.")
        return False
    try:
        print_confirmation(system.book(ev_id, options[pick - 1], start, end, arrival))
        return True
    except ValueError as err:       # includes ConflictError (re-checked inside book)
        print(f"  ! {err}")
        return False


def offer_alternative_times(system, ev_id, preferred, arrival, start, end):
    """Return True if a reservation was made at an alternative time."""
    order = [preferred] + [s for s in sorted(system.stations) if s != preferred]
    slots = system.suggest_time_slots(order, start, end, min_start=arrival)
    if not slots:
        print("  No alternative time slot was found.")
        return False
    print("  Available alternative time slots (same length):")
    for i, (sid, s, e) in enumerate(slots, 1):
        print(f"   {i}. Station {sid}  {format_time(s)}-{format_time(e)}")
    pick = ask_choice("Choose a slot (0 to skip): ", len(slots))
    if pick == 0:
        return False
    sid, s, e = slots[pick - 1]
    try:
        print_confirmation(system.book(ev_id, sid, s, e, arrival))
        return True
    except ValueError as err:
        print(f"  ! {err}")
        return False


def send_to_waiting_list(system, ev_id, stations, arrival, start, end):
    """Put the request on the waiting list (no further question asked).
    stations=None means 'any station'."""
    try:
        system.add_waiting(ev_id, stations, start, end, arrival)
        where = "any station" if stations is None else ", ".join(stations)
        print(f"  You have been placed on the WAITING LIST (accepting: {where}).")
        print("  This is NOT a confirmed reservation yet; you will be allocated")
        print("  automatically if a suitable slot is released.")
    except ValueError as err:
        print(f"  ! {err}")


def offer_waiting_list(system, ev_id, preferred, arrival, start, end):
    """Used by the menu options where the user is browsing: asks before joining."""
    if not ask_yes_no("  Join the waiting list?"):
        return
    any_station = ask_yes_no("  Accept ANY station (otherwise only " + preferred + ")?")
    send_to_waiting_list(system, ev_id, None if any_station else [preferred],
                         arrival, start, end)


def user_view_stations(system):
    print("\n-- Available Charging Stations --")
    print_stations(system)


def user_check_availability(system):
    print("\n-- Check Station Availability --")
    req = ask_request(system, new_ev=False)
    if req is None:
        return
    _, sid, _, start, end = req
    if system.is_free(sid, start, end):
        print(f"  Station {sid} IS available for {format_time(start)}-{format_time(end)}.")
    else:
        print(f"  Station {sid} is NOT available for {format_time(start)}-{format_time(end)}.")


def user_book(system):
    print("\n-- Book Charging Slot --")
    req = ask_request(system)
    if req is None:
        return
    ev_id, sid, arrival, start, end = req
    try:
        print_confirmation(system.book(ev_id, sid, start, end, arrival))
        return
    except ConflictError:
        print(f"  Station {sid} is unavailable for that interval. Looking at other options...")
    except ValueError as err:
        print(f"  ! {err}")
        return
    # Fallback chain: another station (only with permission) -> waiting list
    others_free = bool(system.free_stations(start, end, exclude=sid))
    if offer_alternative_stations(system, ev_id, sid, arrival, start, end):
        return
    # User declined the redirect -> wait for the preferred station only.
    # Nothing else was free -> accept any station that frees up.
    send_to_waiting_list(system, ev_id, [sid] if others_free else None, arrival, start, end)


def user_find_alt_station(system):
    print("\n-- Find Alternative Station --")
    req = ask_request(system)
    if req is None:
        return
    ev_id, sid, arrival, start, end = req
    if not offer_alternative_stations(system, ev_id, sid, arrival, start, end):
        print("  Nothing booked.")


def user_find_alt_time(system):
    print("\n-- Find Alternative Time Slot --")
    req = ask_request(system)
    if req is None:
        return
    ev_id, sid, arrival, start, end = req
    if not offer_alternative_times(system, ev_id, sid, arrival, start, end):
        offer_waiting_list(system, ev_id, sid, arrival, start, end)


def user_join_waiting(system):
    print("\n-- Join Waiting List --")
    req = ask_request(system)
    if req is None:
        return
    ev_id, sid, arrival, start, end = req
    free = system.free_stations(start, end)
    if free:
        print(f"  A slot is free right now at: {', '.join(free)}. Please use 'Book Charging Slot' instead.")
        return
    offer_waiting_list(system, ev_id, sid, arrival, start, end)


def user_view_details(system):
    print("\n-- View Reservation Details --")
    ev_id = ask("Your EV ID: ").upper()
    rid = ask("Reservation ID (blank = show all of my reservations): ").upper()
    rows = system.find_by_ev(ev_id)
    if rid:
        rows = [r for r in rows if r.reservation_id == rid]     # only the owner's reservation
    if not rows:
        print("  No matching reservation found for this EV.")
    else:
        print_reservations(rows)
        for r in rows:
            st = system.stations[r.station_id]
            print(f"  {r.reservation_id}: {st.name}, {st.location}")
    mine = [e for e in system.waiting if e.ev_id == ev_id]
    if mine:
        print("  Your waiting-list requests (not confirmed):")
        for e in mine:
            print(f"   {format_time(e.start)}-{format_time(e.end)}")


def owned_reservation(system, ev_id, rid):
    """Return the reservation only if it belongs to this EV."""
    res = system.find_by_id(rid)
    if res is None or res.ev_id != ev_id.strip().upper():
        return None
    return res


def user_cancel(system):
    print("\n-- Cancel Reservation --")
    ev_id = ask("Your EV ID: ")
    res = owned_reservation(system, ev_id, ask("Reservation ID: "))
    if res is None:
        print("  ! No such reservation for this EV.")
        return
    if not ask_yes_no(f"Cancel {res.reservation_id} ({res.station_id} "
                      f"{format_time(res.start)}-{format_time(res.end)})?"):
        return
    _, allocated = system.cancel(res.reservation_id)
    print("  Reservation cancelled successfully. The slot is available again.")
    if allocated:
        print("  (A waiting EV has been allocated the released slot.)")


def user_update(system):
    print("\n-- Update Reservation --")
    ev_id = ask("Your EV ID: ")
    res = owned_reservation(system, ev_id, ask("Reservation ID: "))
    if res is None:
        print("  ! No such reservation for this EV.")
        return
    print(f"  Current: {res.station_id} {format_time(res.start)}-{format_time(res.end)}")
    try:
        sid = system.get_station(ask("New station ID (blank = keep): ") or res.station_id).station_id
        start = ask_time("New start (HH:MM): ")
        end = ask_time("New end (HH:MM): ")
        validate_interval(start, end)
        res, _ = system.update(res.reservation_id, sid, start, end)
        print(f"  Reservation {res.reservation_id} updated: {res.station_id} "
              f"{format_time(res.start)}-{format_time(res.end)}")
        return
    except ConflictError:
        print("  That interval is not available. Your ORIGINAL reservation is kept.")
    except ValueError as err:
        print(f"  ! {err}")
        return
    # Offer alternatives (ignoring the reservation's own current slot)
    options = system.free_stations(start, end, ignore_rid=res.reservation_id)
    choices = [(s, start, end) for s in options]
    if not choices:
        order = [sid] + [s for s in sorted(system.stations) if s != sid]
        choices = system.suggest_time_slots(order, start, end, ignore_rid=res.reservation_id)
    if not choices:
        print("  No alternative found. Nothing changed.")
        return
    print("  Alternatives:")
    for i, (s, a, b) in enumerate(choices, 1):
        print(f"   {i}. Station {s}  {format_time(a)}-{format_time(b)}")
    pick = ask_choice("Choose (0 to keep original): ", len(choices))
    if pick:
        s, a, b = choices[pick - 1]
        try:
            res, _ = system.update(res.reservation_id, s, a, b)
            print(f"  Reservation {res.reservation_id} updated: {s} {format_time(a)}-{format_time(b)}")
        except ValueError as err:
            print(f"  ! {err} Original kept.")


def user_leave_waiting(system):
    print("\n-- Leave Waiting List --")
    ev_id = ask("Your EV ID: ").upper()
    mine = [e for e in system.waiting if e.ev_id == ev_id]
    if not mine:
        print("  You have no waiting-list requests.")
        return
    for i, e in enumerate(mine, 1):
        stations = "ANY" if e.stations is None else ",".join(e.stations)
        print(f"   {i}. {format_time(e.start)}-{format_time(e.end)}  (stations: {stations})")
    pick = ask_choice("Remove which request (0 to keep all): ", len(mine))
    if pick:
        e = mine[pick - 1]
        system.remove_waiting(e.ev_id, e.start, e.end)
        print("  Removed from the waiting list.")


def user_waiting_time(system):
    print("\n-- Calculate Waiting Time --")
    try:
        arrival = ask_time("Arrival time (HH:MM): ")
        actual = ask_time("Actual charging start (HH:MM): ")
        print(f"  Waiting time = {format_time(actual)} - {format_time(arrival)} = "
              f"{waiting_time(arrival, actual)} minutes")
    except ValueError as err:
        print(f"  ! {err}")


def user_menu(system):
    # user_find_alt_station, user_find_alt_time and user_join_waiting stay in the
    # program but are intentionally not listed in the menu.
    actions = {
        "1": user_view_stations, "2": user_check_availability, "3": user_book,
        "4": user_view_details, "5": user_cancel, "6": user_update,
        "7": user_waiting_time, "8": user_leave_waiting,
    }
    while True:
        print("\n===== USER MENU =====")
        print("1. View Charging Stations\n2. Check Station Availability\n3. Book Charging Slot")
        print("4. View Reservation Details\n5. Cancel Reservation\n6. Update Reservation")
        print("7. Calculate Waiting Time\n8. Leave Waiting List")
        print("9. Back to Main Menu")
        choice = ask("Choose an option: ")
        if choice == "9":
            return
        if choice in actions:
            actions[choice](system)
        else:
            print("  ! Invalid option.")


# ======================================================================
# 7. MAIN
# ======================================================================
def load_demo_data(system):
    system.add_station("S1", "Downtown Hub", "City Centre", 50)
    system.add_station("S2", "Airport Point", "Airport Road", 120)
    system.add_station("S3", "Mall Charger", "Green Mall", 22)
    system.book("EV101", "S1", parse_time("09:00"), parse_time("10:30"), parse_time("08:50"))
    system.book("EV102", "S1", parse_time("10:30"), parse_time("12:00"), parse_time("10:15"))
    system.book("EV103", "S2", parse_time("09:00"), parse_time("11:00"), parse_time("08:45"))


def main():
    system = ChargingSystem()
    if "--demo" in sys.argv:
        load_demo_data(system)
        print("(Demo data loaded: stations S1-S3 and a few reservations.)")
    print("=== EV Charging Station Allocation & Conflict Management ===")
    try:
        while True:
            print("\n===== MAIN MENU =====")
            print("1. Admin Login\n2. User Access\n3. Exit")
            choice = ask("Choose an option: ")
            if choice == "1":
                admin_menu(system)
            elif choice == "2":
                user_menu(system)
            elif choice == "3":
                print("Goodbye!")
                return
            else:
                print("  ! Invalid option.")
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye!")


if __name__ == "__main__":
    main()
