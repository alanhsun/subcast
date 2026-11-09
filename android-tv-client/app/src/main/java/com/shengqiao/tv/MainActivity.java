package com.shengqiao.tv;

import android.app.Activity;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.provider.Settings;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;

/**
 * TV Remote (D-Pad) friendly configuration screen for 雷鸟 TV.
 */
public class MainActivity extends Activity {

    private EditText etServerHost;
    private TextView tvStatus;
    private TextView tvSettingsSummary;
    private final Handler handler = new Handler(Looper.getMainLooper());

    private final Runnable statusRefreshRunnable = new Runnable() {
        @Override
        public void run() {
            updateStatusUi();
            handler.postDelayed(this, 1000);
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        buildTvUi();
        ensureOverlayPermissionAndStartService();
    }

    @Override
    protected void onResume() {
        super.onResume();
        handler.post(statusRefreshRunnable);
    }

    @Override
    protected void onPause() {
        super.onPause();
        handler.removeCallbacks(statusRefreshRunnable);
    }

    private void buildTvUi() {
        SharedPreferences prefs = getSharedPreferences(SubtitleOverlayService.PREFS_NAME, MODE_PRIVATE);

        ScrollView scroll = new ScrollView(this);
        scroll.setBackgroundColor(Color.parseColor("#09090B"));
        scroll.setFillViewport(true);

        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        int pad = dp(36);
        root.setPadding(pad, dp(28), pad, dp(28));

        TextView title = new TextView(this);
        title.setText("声幕 SubCast");
        title.setTextColor(Color.parseColor("#F4F4F5"));
        title.setTextSize(TypedValue.COMPLEX_UNIT_SP, 24f);
        title.setTypeface(Typeface.DEFAULT_BOLD);
        title.setPadding(0, 0, 0, dp(14));
        root.addView(title);

        tvStatus = new TextView(this);
        tvStatus.setTextSize(TypedValue.COMPLEX_UNIT_SP, 15f);
        tvStatus.setTextColor(Color.parseColor("#10B981"));
        tvStatus.setPadding(dp(14), dp(10), dp(14), dp(10));
        GradientDrawable statusBg = new GradientDrawable();
        statusBg.setColor(Color.parseColor("#121215"));
        statusBg.setStroke(dp(1), Color.parseColor("#242429"));
        statusBg.setCornerRadius(dp(6));
        tvStatus.setBackground(statusBg);
        root.addView(tvStatus);

        // Server Host Row
        LinearLayout hostRow = new LinearLayout(this);
        hostRow.setOrientation(LinearLayout.HORIZONTAL);
        hostRow.setGravity(Gravity.CENTER_VERTICAL);
        hostRow.setPadding(0, dp(16), 0, dp(10));

        TextView lblHost = new TextView(this);
        lblHost.setText("网关地址  ");
        lblHost.setTextColor(Color.parseColor("#8E8E96"));
        lblHost.setTextSize(TypedValue.COMPLEX_UNIT_SP, 15f);
        hostRow.addView(lblHost);

        etServerHost = new EditText(this);
        etServerHost.setText(prefs.getString(SubtitleOverlayService.KEY_SERVER_HOST, "192.168.31.23:8765"));
        etServerHost.setTextColor(Color.parseColor("#F4F4F5"));
        etServerHost.setBackgroundColor(Color.parseColor("#121215"));
        etServerHost.setPadding(dp(12), dp(8), dp(12), dp(8));
        etServerHost.setSingleLine(true);
        LinearLayout.LayoutParams etLp = new LinearLayout.LayoutParams(dp(240), LinearLayout.LayoutParams.WRAP_CONTENT);
        etLp.rightMargin = dp(10);
        hostRow.addView(etServerHost, etLp);

        Button btnSaveHost = makeTvButton("连接", v -> {
            String host = etServerHost.getText().toString().trim();
            prefs.edit().putString(SubtitleOverlayService.KEY_SERVER_HOST, host).apply();
            sendServiceAction(SubtitleOverlayService.ACTION_RELOAD_CONFIG);
            Toast.makeText(this, "正在连接 " + host, Toast.LENGTH_SHORT).show();
        });
        hostRow.addView(btnSaveHost);

        Button btnAutoDiscover = makeTvButton("自动发现", v -> {
            sendServiceAction(SubtitleOverlayService.ACTION_DISCOVER_GATEWAY);
            Toast.makeText(this, "正在搜索局域网网关...", Toast.LENGTH_SHORT).show();
        });
        hostRow.addView(btnAutoDiscover);

        root.addView(hostRow);

        // Settings Summary
        tvSettingsSummary = new TextView(this);
        tvSettingsSummary.setTextColor(Color.parseColor("#8E8E96"));
        tvSettingsSummary.setTextSize(TypedValue.COMPLEX_UNIT_SP, 14f);
        tvSettingsSummary.setPadding(0, dp(6), 0, dp(10));
        root.addView(tvSettingsSummary);

        // Controls Row 1: Display mode & Font size
        LinearLayout ctrlRow1 = new LinearLayout(this);
        ctrlRow1.setOrientation(LinearLayout.HORIZONTAL);
        ctrlRow1.setPadding(0, dp(4), 0, dp(8));

        ctrlRow1.addView(makeTvButton("双语 / 仅中文", v -> {
            String cur = prefs.getString(SubtitleOverlayService.KEY_DISPLAY_MODE, "bilingual");
            String next = "bilingual".equals(cur) ? "zh_only" : "bilingual";
            prefs.edit().putString(SubtitleOverlayService.KEY_DISPLAY_MODE, next).apply();
            sendServiceAction(SubtitleOverlayService.ACTION_SEND_SETTINGS);
            updateStatusUi();
        }));

        ctrlRow1.addView(makeTvButton("字号 1.0x", v -> {
            prefs.edit().putFloat(SubtitleOverlayService.KEY_FONT_SCALE, 1.0f).apply();
            sendServiceAction(SubtitleOverlayService.ACTION_SEND_SETTINGS);
            updateStatusUi();
        }));

        ctrlRow1.addView(makeTvButton("字号 1.25x", v -> {
            prefs.edit().putFloat(SubtitleOverlayService.KEY_FONT_SCALE, 1.25f).apply();
            sendServiceAction(SubtitleOverlayService.ACTION_SEND_SETTINGS);
            updateStatusUi();
        }));

        ctrlRow1.addView(makeTvButton("字号 1.5x", v -> {
            prefs.edit().putFloat(SubtitleOverlayService.KEY_FONT_SCALE, 1.5f).apply();
            sendServiceAction(SubtitleOverlayService.ACTION_SEND_SETTINGS);
            updateStatusUi();
        }));

        root.addView(ctrlRow1);

        // Controls Row 2: Delay sync & Test subtitle
        LinearLayout ctrlRow2 = new LinearLayout(this);
        ctrlRow2.setOrientation(LinearLayout.HORIZONTAL);
        ctrlRow2.setPadding(0, dp(4), 0, dp(12));

        ctrlRow2.addView(makeTvButton("提前 0.3s", v -> {
            int cur = prefs.getInt(SubtitleOverlayService.KEY_DELAY_MS, 800);
            int next = Math.max(0, cur - 300);
            prefs.edit().putInt(SubtitleOverlayService.KEY_DELAY_MS, next).apply();
            sendServiceAction(SubtitleOverlayService.ACTION_SEND_SETTINGS);
            updateStatusUi();
        }));

        ctrlRow2.addView(makeTvButton("延后 0.3s", v -> {
            int cur = prefs.getInt(SubtitleOverlayService.KEY_DELAY_MS, 800);
            int next = Math.min(6000, cur + 300);
            prefs.edit().putInt(SubtitleOverlayService.KEY_DELAY_MS, next).apply();
            sendServiceAction(SubtitleOverlayService.ACTION_SEND_SETTINGS);
            updateStatusUi();
        }));

        ctrlRow2.addView(makeTvButton("发送测试", v -> triggerTestSubtitle()));

        ctrlRow2.addView(makeTvButton("后台运行", v -> moveTaskToBack(true)));

        root.addView(ctrlRow2);

        scroll.addView(root);
        setContentView(scroll);
        updateStatusUi();
    }

    private Button makeTvButton(String text, View.OnClickListener onClick) {
        Button btn = new Button(this);
        btn.setText(text);
        btn.setAllCaps(false);
        btn.setTextColor(Color.parseColor("#F4F4F5"));
        btn.setTextSize(TypedValue.COMPLEX_UNIT_SP, 14f);
        btn.setFocusable(true);
        btn.setFocusableInTouchMode(false);
        btn.setPadding(dp(16), dp(9), dp(16), dp(9));

        final GradientDrawable normalBg = new GradientDrawable();
        normalBg.setColor(Color.parseColor("#18181C"));
        normalBg.setStroke(dp(1), Color.parseColor("#242429"));
        normalBg.setCornerRadius(dp(6));

        final GradientDrawable focusBg = new GradientDrawable();
        focusBg.setColor(Color.parseColor("#27272A"));
        focusBg.setCornerRadius(dp(6));
        focusBg.setStroke(dp(2), Color.parseColor("#F4F4F5"));

        btn.setBackground(normalBg);
        btn.setOnFocusChangeListener((v, hasFocus) -> v.setBackground(hasFocus ? focusBg : normalBg));
        btn.setOnClickListener(onClick);

        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
        );
        lp.rightMargin = dp(10);
        btn.setLayoutParams(lp);
        return btn;
    }

    private void updateStatusUi() {
        SharedPreferences prefs = getSharedPreferences(SubtitleOverlayService.PREFS_NAME, MODE_PRIVATE);
        String host = prefs.getString(SubtitleOverlayService.KEY_SERVER_HOST, "192.168.31.23:8765");
        if (!etServerHost.hasFocus() && !host.equals(etServerHost.getText().toString())) {
            etServerHost.setText(host);
        }
        tvStatus.setText(SubtitleOverlayService.currentStatusText);
        tvStatus.setTextColor(
                SubtitleOverlayService.isConnected
                        ? Color.parseColor("#10B981")
                        : Color.parseColor("#F59E0B")
        );

        String mode = "zh_only".equals(prefs.getString(SubtitleOverlayService.KEY_DISPLAY_MODE, "bilingual"))
                ? "仅中文"
                : "双语对照";
        float scale = prefs.getFloat(SubtitleOverlayService.KEY_FONT_SCALE, 1.0f);
        int delayMs = prefs.getInt(SubtitleOverlayService.KEY_DELAY_MS, 800);
        tvSettingsSummary.setText(
                String.format("显示: %s   |   字号: %.2fx   |   偏移: %.1fs", mode, scale, delayMs / 1000f)
        );
    }

    private void ensureOverlayPermissionAndStartService() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M && !Settings.canDrawOverlays(this)) {
            Toast.makeText(
                    this,
                    "提示：请允许悬浮窗权限，或运行电脑端 install_to_tv.ps1 自动授权",
                    Toast.LENGTH_LONG
            ).show();
            try {
                Intent intent = new Intent(
                        Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                        Uri.parse("package:" + getPackageName())
                );
                startActivity(intent);
            } catch (Exception ignored) {
                // Domestic TV ROMs often strip this settings activity; ADB appops handles it
            }
        }
        sendServiceAction(SubtitleOverlayService.ACTION_RELOAD_CONFIG);
    }

    private void sendServiceAction(String action) {
        Intent svc = new Intent(this, SubtitleOverlayService.class);
        svc.setAction(action);
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                startForegroundService(svc);
            } else {
                startService(svc);
            }
        } catch (Exception e) {
            startService(svc);
        }
    }

    private void triggerTestSubtitle() {
        SharedPreferences prefs = getSharedPreferences(SubtitleOverlayService.PREFS_NAME, MODE_PRIVATE);
        final String host = prefs.getString(SubtitleOverlayService.KEY_SERVER_HOST, "192.168.31.23:8765");
        Thread t = new Thread(() -> {
            try {
                URL url = new URL("http://" + host + "/api/test-subtitle");
                HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                conn.setRequestMethod("POST");
                conn.setConnectTimeout(4000);
                conn.setDoOutput(true);
                conn.setRequestProperty("Content-Type", "application/json");
                String body = "{\"en\":\"Good evening, live English broadcast translation is working on your TV.\",\"zh\":\"\"}";
                try (OutputStream os = conn.getOutputStream()) {
                    os.write(body.getBytes(StandardCharsets.UTF_8));
                }
                conn.getResponseCode();
                conn.disconnect();
            } catch (Exception e) {
                handler.post(() -> Toast.makeText(
                        MainActivity.this,
                        "发送测试请求失败: " + e.getMessage(),
                        Toast.LENGTH_SHORT
                ).show());
            }
        });
        t.setDaemon(true);
        t.start();
    }

    private int dp(int value) {
        return Math.round(
                TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP,
                        value,
                        getResources().getDisplayMetrics()
                )
        );
    }
}
