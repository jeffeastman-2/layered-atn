# atn/core.py
from latn.lexer.token_stream import TokenStream
from latn.utils.debug import debug_print

class ATNState:
    def __init__(self, name):
        self.name = name
        self.arcs = []

    def add_arc(self, test_fn, action_fn, next_state):
        self.arcs.append((test_fn, action_fn, next_state))

    def __repr__(self):
        return f"ATNState({self.name})"

def noop(_, tok):
    pass

# Walk the whole ATN to get all states
def get_all_states(start_state):
    visited = set()
    stack = [start_state]
    all_states = []

    while stack:
        state = stack.pop()
        if id(state) in visited:  # Use id to avoid hashing issues
            continue
        visited.add(id(state))
        all_states.append(state)

        for _, _, next_state in state.arcs:
            if isinstance(next_state, ATNState):
                stack.append(next_state)
            else:
                debug_print(f"Warning: next_state {next_state} is not an ATNState")
    return all_states

def run_atn(start_state, end_state, ts: TokenStream, pos):
    current = start_state

    while True:
        tok = ts.peek()
        matched = False

        for test, action, next_state in current.arcs:
            if tok is not None:
                debug_print(f"🔍  Testing '{tok.word}' in {current.name} → {next_state.name}")
            else:
                debug_print(f"🔍  Testing 'None' in {current.name} → {next_state.name}")
                
            match_result = test(tok)
            if isinstance(match_result, tuple):
                matched_bool, should_consume = match_result
            else:
                matched_bool, should_consume = match_result, True

            if matched_bool:
                if tok is not None:
                    debug_print(f"    🎯  Token '{tok.word}' matches in {current.name} → {next_state.name}")
                else:
                    debug_print(f"    🎯  Token is None, but matched in {current.name} → {next_state.name}")

                if action is None:
                    debug_print(" ⚠️ ERROR: This arc has a None action!")

                action(pos, tok)
                current = next_state

                # Advance the token stream if needed
                if should_consume and tok is not None and action != noop:
                    ts.advance()
                else:
                    debug_print(f"    Token '{tok}' accepted but not consumed") 
                    debug_print(f"        should_consume = {should_consume}")   
                    if tok is None: debug_print(f"        tok was None")   
                    debug_print(f"        action was {action}") 

                matched = True
                break
            else:
                if tok is not None:
                    debug_print(f"    ✗ Failed to match in {current.name} on '{tok.word}' → {next_state.name}")
                else:
                    debug_print(f"    ✗ Failed to match in {current.name} on None → {next_state.name}")

        if not matched:
            if tok is not None:
                debug_print(f"    ⚠️ No arc matched in {current.name} on token '{tok.word}'")
            else:
                debug_print(f"    ⚠️ No arc matched in {current.name} on None token")
            return None

        if current == end_state:
            debug_print(f"✅ Reached final state: {end_state.name} with context: {pos}")
            return pos
