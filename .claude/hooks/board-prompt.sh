#!/usr/bin/env bash
# UserPromptSubmit: surface board posts that arrived since the last check. Silent when none. Never fails.
cd "${CLAUDE_PROJECT_DIR:-.}" 2>/dev/null || exit 0
export BOARD_AGENT="${BOARD_AGENT:-Claude}"
out="$(tools/board unread 2>/dev/null)"
[ -n "$out" ] && printf 'New board posts:\n%s\n' "$out"
exit 0
