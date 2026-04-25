package com.yzamari.turboquant.assistant

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.BatteryManager
import android.provider.AlarmClock
import android.provider.CalendarContract
import org.json.JSONObject
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone

/**
 * Executes a parsed tool call against the Android system.
 *
 * Returns a [ToolResult] containing a short JSON message that the
 * assistant can read back to the user.
 */
class ToolDispatcher(private val context: Context) {

    data class ToolResult(val ok: Boolean, val message: String) {
        fun toJson(): String =
            JSONObject().put("ok", ok).put("message", message).toString()
    }

    fun dispatch(name: String, argsJson: JSONObject): ToolResult = try {
        when (name) {
            "current_time"    -> currentTime()
            "current_battery" -> currentBattery()
            "set_alarm"       -> setAlarm(argsJson)
            "set_timer"       -> setTimer(argsJson)
            "web_search"      -> webSearch(argsJson)
            "open_url"        -> openUrl(argsJson)
            "call"            -> dial(argsJson)
            "sms"             -> sms(argsJson)
            "email"           -> email(argsJson)
            "directions"      -> directions(argsJson)
            "open_app"        -> openApp(argsJson)
            "add_calendar"    -> addCalendar(argsJson)
            else              -> ToolResult(false, "Unknown tool: $name")
        }
    } catch (e: Throwable) {
        ToolResult(false, "Tool '$name' failed: ${e.message}")
    }

    // ---------------------------------------------------------------------

    private fun startNew(intent: Intent) {
        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        context.startActivity(intent)
    }

    private fun currentTime(): ToolResult {
        val fmt = SimpleDateFormat("EEEE, MMMM d, yyyy h:mm a", Locale.getDefault())
        fmt.timeZone = TimeZone.getDefault()
        return ToolResult(true, fmt.format(Date()))
    }

    private fun currentBattery(): ToolResult {
        val bm = context.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
        val level = bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)
        return ToolResult(true, "Battery is at $level%")
    }

    private fun setAlarm(args: JSONObject): ToolResult {
        val time = args.optString("time").trim()
        val parts = time.split(":")
        if (parts.size != 2) return ToolResult(false, "Invalid time '$time' (expected HH:mm).")
        val hour = parts[0].toIntOrNull() ?: return ToolResult(false, "Invalid hour in '$time'.")
        val min  = parts[1].toIntOrNull() ?: return ToolResult(false, "Invalid minute in '$time'.")
        val label = args.optString("label").ifBlank { null }

        val intent = Intent(AlarmClock.ACTION_SET_ALARM).apply {
            putExtra(AlarmClock.EXTRA_HOUR, hour)
            putExtra(AlarmClock.EXTRA_MINUTES, min)
            putExtra(AlarmClock.EXTRA_SKIP_UI, true)
            if (label != null) putExtra(AlarmClock.EXTRA_MESSAGE, label)
        }
        startNew(intent)
        return ToolResult(true, "Alarm set for ${"%02d:%02d".format(hour, min)}" +
            (if (label != null) " ($label)" else ""))
    }

    private fun setTimer(args: JSONObject): ToolResult {
        val seconds = args.optInt("seconds", -1)
        if (seconds <= 0) return ToolResult(false, "Invalid seconds.")
        val label = args.optString("label").ifBlank { null }
        val intent = Intent(AlarmClock.ACTION_SET_TIMER).apply {
            putExtra(AlarmClock.EXTRA_LENGTH, seconds)
            putExtra(AlarmClock.EXTRA_SKIP_UI, true)
            if (label != null) putExtra(AlarmClock.EXTRA_MESSAGE, label)
        }
        startNew(intent)
        return ToolResult(true, "Timer set for ${seconds}s" +
            (if (label != null) " ($label)" else ""))
    }

    private fun webSearch(args: JSONObject): ToolResult {
        val query = args.optString("query")
        if (query.isBlank()) return ToolResult(false, "Empty query.")
        val intent = Intent(Intent.ACTION_WEB_SEARCH).apply {
            putExtra("query", query)
        }
        startNew(intent)
        return ToolResult(true, "Searching the web for: $query")
    }

    private fun openUrl(args: JSONObject): ToolResult {
        val url = args.optString("url")
        if (url.isBlank()) return ToolResult(false, "Empty URL.")
        val intent = Intent(Intent.ACTION_VIEW, Uri.parse(url))
        startNew(intent)
        return ToolResult(true, "Opening $url")
    }

    private fun dial(args: JSONObject): ToolResult {
        val phone = args.optString("phone")
        if (phone.isBlank()) return ToolResult(false, "Empty phone number.")
        val intent = Intent(Intent.ACTION_DIAL, Uri.parse("tel:$phone"))
        startNew(intent)
        return ToolResult(true, "Opening dialer for $phone")
    }

    private fun sms(args: JSONObject): ToolResult {
        val to   = args.optString("to")
        val body = args.optString("body")
        if (to.isBlank()) return ToolResult(false, "Missing recipient.")
        val intent = Intent(Intent.ACTION_SENDTO, Uri.parse("smsto:$to")).apply {
            putExtra("sms_body", body)
        }
        startNew(intent)
        return ToolResult(true, "Composing SMS to $to")
    }

    private fun email(args: JSONObject): ToolResult {
        val to   = args.optString("to")
        if (to.isBlank()) return ToolResult(false, "Missing recipient.")
        val subject = args.optString("subject")
        val body    = args.optString("body")
        val intent = Intent(Intent.ACTION_SENDTO, Uri.parse("mailto:$to")).apply {
            if (subject.isNotBlank()) putExtra(Intent.EXTRA_SUBJECT, subject)
            if (body.isNotBlank())    putExtra(Intent.EXTRA_TEXT, body)
        }
        startNew(intent)
        return ToolResult(true, "Composing email to $to")
    }

    private fun directions(args: JSONObject): ToolResult {
        val dest = args.optString("destination")
        if (dest.isBlank()) return ToolResult(false, "Missing destination.")
        val intent = Intent(
            Intent.ACTION_VIEW,
            Uri.parse("google.navigation:q=" + Uri.encode(dest))
        )
        startNew(intent)
        return ToolResult(true, "Navigating to $dest")
    }

    private fun openApp(args: JSONObject): ToolResult {
        val pkg = args.optString("package_name")
        if (pkg.isBlank()) return ToolResult(false, "Missing package_name.")
        val intent = context.packageManager.getLaunchIntentForPackage(pkg)
            ?: return ToolResult(false, "App $pkg is not installed.")
        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        context.startActivity(intent)
        return ToolResult(true, "Launching $pkg")
    }

    private fun addCalendar(args: JSONObject): ToolResult {
        val title = args.optString("title")
        val whenIso = args.optString("when")
        if (title.isBlank() || whenIso.isBlank())
            return ToolResult(false, "title and when are required.")

        val isoFmts = listOf(
            "yyyy-MM-dd'T'HH:mm:ss",
            "yyyy-MM-dd'T'HH:mm",
            "yyyy-MM-dd HH:mm",
        )
        var startMs: Long? = null
        for (p in isoFmts) {
            try {
                val sdf = SimpleDateFormat(p, Locale.US)
                sdf.timeZone = TimeZone.getDefault()
                startMs = sdf.parse(whenIso)?.time
                if (startMs != null) break
            } catch (_: Throwable) { /* try next */ }
        }
        startMs ?: return ToolResult(false, "Could not parse 'when': $whenIso")

        val durationMin = args.optInt("duration_minutes", 60).coerceAtLeast(1)
        val endMs = startMs + durationMin * 60_000L

        val intent = Intent(Intent.ACTION_INSERT, CalendarContract.Events.CONTENT_URI).apply {
            putExtra(CalendarContract.Events.TITLE, title)
            putExtra(CalendarContract.EXTRA_EVENT_BEGIN_TIME, startMs)
            putExtra(CalendarContract.EXTRA_EVENT_END_TIME, endMs)
        }
        startNew(intent)
        return ToolResult(true, "Adding '$title' to calendar at $whenIso")
    }
}
