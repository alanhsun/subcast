package com.shengqiao.tv;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PixelFormat;
import android.graphics.Typeface;
import android.os.Build;
import android.os.Handler;
import android.os.IBinder;
import android.os.Looper;
import android.text.TextPaint;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.View;
import android.view.WindowManager;
import android.widget.LinearLayout;
import android.widget.TextView;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.nio.charset.StandardCharsets;
import org.json.JSONObject;

/**
 * Foreground overlay service that displays non-focusable bilingual subtitles at the bottom
 * of the TV screen while watching 影视仓 (TVBox) or Jellyfin live channels.
 */
public class SubtitleOverlayService extends Service {

    public static final String PREFS_NAME = "shengqiao_tv_prefs";
    public static final String KEY_SERVER_HOST = "server_host";
    public static final String KEY_DISPLAY_MODE = "display_mode";
    public static final String KEY_FONT_SCALE = "font_scale";
    public static final String KEY_DELAY_MS = "delay_ms";

    public static final String ACTION_RELOAD_CONFIG = "com.shengqiao.tv.RELOAD_CONFIG";
    public static final String ACTION_SEND_SETTINGS = "com.shengqiao.tv.SEND_SETTINGS";
    public static final String ACTION_DISCOVER_GATEWAY = "com.shengqiao.tv.DISCOVER_GATEWAY";

    public static volatile String currentStatusText = "未连接";
    public static volatile boolean isConnected = false;

    private WindowManager windowManager;
    private LinearLayout overlayContainer;
    private TextView tvEn;
    private TextView tvZh;
    private final Handler mainHandler = new Handler(Looper.getMainLooper());
    private SimpleWebSocketClient wsClient;
    private volatile boolean serviceRunning = false;

    private final Runnable autoHideRunnable = new Runnable() {
        @Override
        public void run() {
            if (overlayContainer != null) {
                overlayContainer.setVisibility(View.GONE);
            }
        }
    };

    private final Runnable reconnectRunnable = new Runnable() {
        @Override
        public void run() {
            if (serviceRunning) {
                connectWebSocket();
            }
        }
    };

    @Override
    public void onCreate() {
        super.onCreate();
        serviceRunning = true;
        startForegroundCompat();
        createOverlayView();
        connectWebSocket();
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent != null && intent.getAction() != null) {
            String act = intent.getAction();
            if (ACTION_RELOAD_CONFIG.equals(act)) {
                applyOverlayStyles();
                connectWebSocket();
            } else if (ACTION_SEND_SETTINGS.equals(act)) {
                applyOverlayStyles();
                pushSettingsToServer();
            } else if (ACTION_DISCOVER_GATEWAY.equals(act)) {
                discoverGatewayViaUdp();
            }
        }
        return START_STICKY;
    }

    @Override
    public void onDestroy() {
        serviceRunning = false;
        mainHandler.removeCallbacksAndMessages(null);
        if (wsClient != null) {
            wsClient.close();
        }
        if (windowManager != null && overlayContainer != null) {
            try {
                windowManager.removeView(overlayContainer);
            } catch (Exception ignored) {
            }
        }
        super.onDestroy();
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    private void startForegroundCompat() {
        String channelId = "shengqiao_tv_overlay";
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            NotificationChannel chan = new NotificationChannel(
                    channelId,
                    "声桥 TV 实时字幕服务",
                    NotificationManager.IMPORTANCE_LOW
            );
            NotificationManager nm = (NotificationManager) getSystemService(Context.NOTIFICATION_SERVICE);
            if (nm != null) {
                nm.createNotificationChannel(chan);
            }
            Notification notif = new Notification.Builder(this, channelId)
                    .setContentTitle("声桥 TV 实时字幕运行中")
                    .setContentText("正在监听直播流双语字幕")
                    .setSmallIcon(android.R.drawable.ic_btn_speak_now)
                    .build();
            startForeground(1001, notif);
        } else {
            Notification notif = new Notification.Builder(this)
                    .setContentTitle("声桥 TV 实时字幕运行中")
                    .setContentText("正在监听直播流双语字幕")
                    .setSmallIcon(android.R.drawable.ic_btn_speak_now)
                    .build();
            startForeground(1001, notif);
        }
    }

    private void createOverlayView() {
        windowManager = (WindowManager) getSystemService(WINDOW_SERVICE);

        overlayContainer = new LinearLayout(this);
        overlayContainer.setOrientation(LinearLayout.VERTICAL);
        overlayContainer.setGravity(Gravity.CENTER_HORIZONTAL);
        int padH = dp(16);
        int padV = dp(4);
        overlayContainer.setPadding(padH, padV, padH, padV);
        // Transparent background so subtitles never obscure the video screen
        overlayContainer.setBackground(null);

        int maxW = (int) (getResources().getDisplayMetrics().widthPixels * 0.90f);

        tvEn = new StrokeTextView(this, dp(3));
        tvEn.setGravity(Gravity.CENTER);
        tvEn.setTextColor(Color.parseColor("#F8FAFC"));
        tvEn.setShadowLayer(6f, 0f, 2f, Color.BLACK);
        tvEn.setLineSpacing(dp(2), 1.12f);
        tvEn.setMaxWidth(maxW);
        tvEn.setMaxLines(4);

        tvZh = new StrokeTextView(this, dp(4));
        tvZh.setGravity(Gravity.CENTER);
        tvZh.setTextColor(Color.parseColor("#FDE047"));
        tvZh.setTypeface(Typeface.DEFAULT_BOLD);
        tvZh.setShadowLayer(8f, 0f, 3f, Color.BLACK);
        tvZh.setLineSpacing(dp(2), 1.12f);
        tvZh.setMaxWidth(maxW);
        tvZh.setMaxLines(4);

        overlayContainer.addView(tvEn);
        overlayContainer.addView(tvZh);
        overlayContainer.setVisibility(View.GONE);

        applyOverlayStyles();

        int layoutType = (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O)
                ? WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY
                : WindowManager.LayoutParams.TYPE_SYSTEM_ALERT;

        WindowManager.LayoutParams params = new WindowManager.LayoutParams(
                WindowManager.LayoutParams.WRAP_CONTENT,
                WindowManager.LayoutParams.WRAP_CONTENT,
                layoutType,
                WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE
                        | WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE
                        | WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN
                        | WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON,
                PixelFormat.TRANSLUCENT
        );
        params.gravity = Gravity.BOTTOM | Gravity.CENTER_HORIZONTAL;
        params.y = dp(46);

        try {
            windowManager.addView(overlayContainer, params);
        } catch (Exception e) {
            currentStatusText = "悬浮窗权限未授予: " + e.getMessage();
        }
    }

    private void applyOverlayStyles() {
        SharedPreferences prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE);
        float scale = prefs.getFloat(KEY_FONT_SCALE, 1.0f);
        String mode = prefs.getString(KEY_DISPLAY_MODE, "bilingual");
        if (tvEn != null && tvZh != null) {
            tvEn.setTextSize(TypedValue.COMPLEX_UNIT_SP, 18f * scale);
            tvZh.setTextSize(TypedValue.COMPLEX_UNIT_SP, 26f * scale);
            tvEn.setVisibility("zh_only".equals(mode) ? View.GONE : View.VISIBLE);
        }
    }

    private void connectWebSocket() {
        mainHandler.removeCallbacks(reconnectRunnable);
        if (wsClient != null) {
            wsClient.close();
        }

        SharedPreferences prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE);
        String hostPort = prefs.getString(KEY_SERVER_HOST, "192.168.31.23:8765");
        String wsUrl = "ws://" + hostPort.trim() + "/ws/subtitles";
        currentStatusText = "正在连接 " + hostPort + " ...";

        wsClient = new SimpleWebSocketClient(wsUrl, new SimpleWebSocketClient.Listener() {
            @Override
            public void onConnected() {
                isConnected = true;
                currentStatusText = "已连接网关 (" + hostPort + ")";
                mainHandler.post(() -> showSubtitleOnUi(
                        "LinguaRelay-TV Connected",
                        "声桥 TV 字幕已就绪 (" + hostPort + ")",
                        true,
                        4000
                ));
            }

            @Override
            public void onMessage(String text) {
                handleServerJson(text);
            }

            @Override
            public void onDisconnected(String reason) {
                isConnected = false;
                currentStatusText = "连接断开，3秒后重连 (" + hostPort + ")";
                discoverGatewayViaUdp();
                mainHandler.postDelayed(reconnectRunnable, 3000);
            }
        });
        wsClient.connectAsync();
    }

    private void handleServerJson(String rawJson) {
        try {
            JSONObject obj = new JSONObject(rawJson);
            String type = obj.optString("type", "");
            if ("subtitle".equals(type) || "status".equals(type)) {
                final String en = obj.optString("en", "");
                final String zh = obj.optString("zh", "");
                final boolean isFinal = obj.optBoolean("is_final", true);
                final String mode = obj.optString("display_mode", "");

                if (!mode.isEmpty()) {
                    getSharedPreferences(PREFS_NAME, MODE_PRIVATE)
                            .edit()
                            .putString(KEY_DISPLAY_MODE, mode)
                            .apply();
                }

                mainHandler.post(() -> {
                    applyOverlayStyles();
                    // Keep subtitle visible for up to 45s so consecutive subtitles transition seamlessly with zero gap
                    showSubtitleOnUi(en, zh, isFinal, 45000);
                });
            } else if ("clear".equals(type)) {
                mainHandler.post(() -> {
                    if (overlayContainer != null) {
                        overlayContainer.setVisibility(View.GONE);
                    }
                });
            }
        } catch (Exception ignored) {
        }
    }

    private void showSubtitleOnUi(String en, String zh, boolean isFinal, long hideDelayMs) {
        if (overlayContainer == null || tvEn == null || tvZh == null) return;
        if ((en == null || en.isEmpty()) && (zh == null || zh.isEmpty())) return;

        tvEn.setText(en != null ? en : "");
        tvZh.setText(zh != null ? zh : "");
        tvZh.setTextColor(isFinal ? Color.parseColor("#FDE047") : Color.parseColor("#BAE6FD"));

        overlayContainer.setVisibility(View.VISIBLE);
        mainHandler.removeCallbacks(autoHideRunnable);
        mainHandler.postDelayed(autoHideRunnable, hideDelayMs);
    }

    private void pushSettingsToServer() {
        if (wsClient == null || !isConnected) return;
        try {
            SharedPreferences prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE);
            JSONObject msg = new JSONObject();
            msg.put("action", "update_settings");
            msg.put("display_mode", prefs.getString(KEY_DISPLAY_MODE, "bilingual"));
            msg.put("font_scale", prefs.getFloat(KEY_FONT_SCALE, 1.0f));
            msg.put("subtitle_delay_ms", prefs.getInt(KEY_DELAY_MS, 800));
            wsClient.sendText(msg.toString());
        } catch (Exception ignored) {
        }
    }

    private void discoverGatewayViaUdp() {
        Thread t = new Thread(() -> {
            DatagramSocket sock = null;
            try {
                sock = new DatagramSocket();
                sock.setBroadcast(true);
                sock.setSoTimeout(2000);
                byte[] req = "LINGUARELAY_DISCOVER".getBytes(StandardCharsets.UTF_8);

                String[] targets = new String[]{"255.255.255.255", "192.168.31.255", "10.10.10.255", "10.10.10.80"};
                for (String target : targets) {
                    try {
                        DatagramPacket pkt = new DatagramPacket(
                                req, req.length, InetAddress.getByName(target), 8766
                        );
                        sock.send(pkt);
                    } catch (Exception ignored) {
                    }
                }

                byte[] buf = new byte[512];
                DatagramPacket resp = new DatagramPacket(buf, buf.length);
                sock.receive(resp);
                String body = new String(resp.getData(), 0, resp.getLength(), StandardCharsets.UTF_8);
                JSONObject json = new JSONObject(body);
                if ("lingua-relay-tv".equals(json.optString("service"))) {
                    String ip = resp.getAddress().getHostAddress();
                    int port = json.optInt("port", 8765);
                    String foundHost = ip + ":" + port;
                    SharedPreferences prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE);
                    String oldHost = prefs.getString(KEY_SERVER_HOST, "");
                    if (!foundHost.equals(oldHost) || !isConnected) {
                        prefs.edit().putString(KEY_SERVER_HOST, foundHost).apply();
                        mainHandler.post(this::connectWebSocket);
                    }
                }
            } catch (Exception ignored) {
            } finally {
                if (sock != null) {
                    sock.close();
                }
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

    /**
     * Cinema-style outlined TextView that renders a crisp black stroke around every character
     * so subtitles remain 100% legible over bright video scenes without needing a dark background box.
     */
    private static class StrokeTextView extends TextView {
        private final float strokeWidthPx;
        private boolean isDrawing = false;

        public StrokeTextView(Context context, float strokeWidthPx) {
            super(context);
            this.strokeWidthPx = strokeWidthPx;
        }

        @Override
        public void invalidate() {
            if (isDrawing) return;
            super.invalidate();
        }

        @Override
        protected void onDraw(Canvas canvas) {
            CharSequence text = getText();
            if (text != null && text.length() > 0) {
                isDrawing = true;
                int origColor = getCurrentTextColor();
                TextPaint paint = getPaint();

                paint.setStyle(Paint.Style.STROKE);
                paint.setStrokeJoin(Paint.Join.ROUND);
                paint.setStrokeCap(Paint.Cap.ROUND);
                paint.setStrokeMiter(10f);
                paint.setStrokeWidth(strokeWidthPx);
                setTextColor(Color.BLACK);
                super.onDraw(canvas);

                paint.setStyle(Paint.Style.FILL);
                setTextColor(origColor);
                isDrawing = false;
            }
            super.onDraw(canvas);
        }
    }
}
