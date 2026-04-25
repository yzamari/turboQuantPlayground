package com.yzamari.turboquant.assistant

/**
 * Declarative description of the tools available to the assistant.
 *
 * The [systemPrompt] function renders a Llama-3.2 friendly tool-calling
 * preamble. Llama-3.2 is trained to emit tool calls with a leading
 * `<|python_tag|>` token, but for compactness here we instruct the model
 * to emit tool calls as a single JSON object on its own line:
 *
 *   {"tool":"set_alarm","arguments":{"time":"07:00","label":"wake up"}}
 *
 * The orchestrator parses these out of the generated stream.
 */
data class Tool(
    val name: String,
    val description: String,
    val parameters: List<ToolParam>,
)

data class ToolParam(
    val name: String,
    val type: String,
    val description: String,
    val required: Boolean = true,
)

object Tools {
    val all: List<Tool> = listOf(
        Tool(
            name = "current_time",
            description = "Get the current local time on the device. Use this when the user asks for the time.",
            parameters = emptyList(),
        ),
        Tool(
            name = "current_battery",
            description = "Get the current battery level (percent).",
            parameters = emptyList(),
        ),
        Tool(
            name = "set_alarm",
            description = "Schedule an alarm in the system clock app at a specific 24h time.",
            parameters = listOf(
                ToolParam("time", "string", "Time in HH:mm 24-hour format, e.g. '07:00'."),
                ToolParam("label", "string", "Optional label for the alarm.", required = false),
            ),
        ),
        Tool(
            name = "set_timer",
            description = "Start a countdown timer for the given number of seconds.",
            parameters = listOf(
                ToolParam("seconds", "integer", "Total seconds for the timer."),
                ToolParam("label", "string", "Optional label.", required = false),
            ),
        ),
        Tool(
            name = "web_search",
            description = "Open a web search for the given query in the system browser.",
            parameters = listOf(
                ToolParam("query", "string", "What to search for."),
            ),
        ),
        Tool(
            name = "open_url",
            description = "Open the given URL in the user's browser.",
            parameters = listOf(
                ToolParam("url", "string", "Full URL beginning with http(s)://"),
            ),
        ),
        Tool(
            name = "call",
            description = "Open the dialer pre-filled with the given phone number (does not place the call automatically).",
            parameters = listOf(
                ToolParam("phone", "string", "Phone number including country code if needed."),
            ),
        ),
        Tool(
            name = "sms",
            description = "Compose an SMS to the given recipient with a pre-filled body.",
            parameters = listOf(
                ToolParam("to", "string", "Recipient phone number."),
                ToolParam("body", "string", "Message text."),
            ),
        ),
        Tool(
            name = "email",
            description = "Compose an email to the given recipient.",
            parameters = listOf(
                ToolParam("to", "string", "Recipient email address."),
                ToolParam("subject", "string", "Email subject.", required = false),
                ToolParam("body", "string", "Email body.", required = false),
            ),
        ),
        Tool(
            name = "directions",
            description = "Open Google Maps with directions to the given destination.",
            parameters = listOf(
                ToolParam("destination", "string", "Address or place name."),
            ),
        ),
        Tool(
            name = "open_app",
            description = "Launch an installed app by its package name.",
            parameters = listOf(
                ToolParam("package_name", "string", "e.g. 'com.spotify.music'."),
            ),
        ),
        Tool(
            name = "add_calendar",
            description = "Open the calendar with a pre-filled new-event form.",
            parameters = listOf(
                ToolParam("title", "string", "Event title."),
                ToolParam("when", "string", "ISO-8601 start time, e.g. '2026-04-26T18:00'."),
                ToolParam(
                    "duration_minutes",
                    "integer",
                    "Optional duration in minutes (default 60).",
                    required = false,
                ),
            ),
        ),
    )

    /** Render a Llama-3.2-friendly system prompt that primes tool calling. */
    fun systemPrompt(): String = buildString {
        appendLine("You are a concise on-device personal assistant running on a phone.")
        appendLine("You have access to a set of tools. When a tool would help the user,")
        appendLine("emit a single JSON object on its OWN LINE in this exact format:")
        appendLine("    {\"tool\":\"<tool_name>\",\"arguments\":{...}}")
        appendLine("After the tool runs the user will provide its result; then reply naturally")
        appendLine("in plain text. Do NOT emit JSON unless calling a tool. Keep answers short.")
        appendLine()
        appendLine("Available tools:")
        for (t in all) {
            append("- ")
            append(t.name)
            append("(")
            append(t.parameters.joinToString(", ") {
                val req = if (it.required) "" else "?"
                "${it.name}${req}: ${it.type}"
            })
            append("): ")
            appendLine(t.description)
        }
        appendLine()
        appendLine("If no tool is needed, just answer directly.")
    }
}
