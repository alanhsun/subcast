package com.shengqiao.tv;

import android.util.Base64;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.security.SecureRandom;

/**
 * Lightweight zero-dependency RFC-6455 WebSocket client for Android TV.
 */
public class SimpleWebSocketClient {

    public interface Listener {
        void onConnected();
        void onMessage(String text);
        void onDisconnected(String reason);
    }

    private final String wsUrl;
    private final Listener listener;
    private volatile boolean running = false;
    private Socket socket;
    private OutputStream outStream;
    private final SecureRandom random = new SecureRandom();
    private final Object writeLock = new Object();

    public SimpleWebSocketClient(String wsUrl, Listener listener) {
        this.wsUrl = wsUrl;
        this.listener = listener;
    }

    public void connectAsync() {
        running = true;
        Thread t = new Thread(this::runLoop, "LinguaRelay-WS");
        t.setDaemon(true);
        t.start();
    }

    public void close() {
        running = false;
        try {
            if (socket != null) {
                socket.close();
            }
        } catch ( Exception ignored) {
        }
    }

    public void sendText(String text) {
        Thread t = new Thread(() -> {
            try {
                byte[] payload = text.getBytes(StandardCharsets.UTF_8);
                sendFrame(0x1, payload);
            } catch (Exception ignored) {
            }
        });
        t.setDaemon(true);
        t.start();
    }

    private void runLoop() {
        try {
            URI uri = new URI(wsUrl);
            String host = uri.getHost();
            int port = uri.getPort() > 0 ? uri.getPort() : 8765;
            String path = (uri.getRawPath() != null && !uri.getRawPath().isEmpty())
                    ? uri.getRawPath()
                    : "/ws/subtitles";
            if (uri.getRawQuery() != null && !uri.getRawQuery().isEmpty()) {
                path += "?" + uri.getRawQuery();
            }

            socket = new Socket();
            socket.connect(new InetSocketAddress(host, port), 5000);
            socket.setSoTimeout(45000);
            socket.setTcpNoDelay(true);

            InputStream rawIn = new BufferedInputStream(socket.getInputStream());
            outStream = new BufferedOutputStream(socket.getOutputStream());

            byte[] keyBytes = new byte[16];
            random.nextBytes(keyBytes);
            String secKey = Base64.encodeToString(keyBytes, Base64.NO_WRAP);

            String handshake = "GET " + path + " HTTP/1.1\r\n"
                    + "Host: " + host + ":" + port + "\r\n"
                    + "Upgrade: websocket\r\n"
                    + "Connection: Upgrade\r\n"
                    + "Sec-WebSocket-Key: " + secKey + "\r\n"
                    + "Sec-WebSocket-Version: 13\r\n\r\n";
            outStream.write(handshake.getBytes(StandardCharsets.US_ASCII));
            outStream.flush();

            String statusLine = readAsciiLine(rawIn);
            if (statusLine == null || !statusLine.contains("101")) {
                throw new IOException("WebSocket handshake failed: " + statusLine);
            }
            while (true) {
                String headerLine = readAsciiLine(rawIn);
                if (headerLine == null || headerLine.isEmpty()) {
                    break;
                }
            }

            listener.onConnected();
            DataInputStream dis = new DataInputStream(rawIn);

            while (running) {
                int b0 = dis.readUnsignedByte();
                int b1 = dis.readUnsignedByte();
                int opcode = b0 & 0x0F;
                boolean masked = (b1 & 0x80) != 0;
                long payloadLen = b1 & 0x7F;

                if (payloadLen == 126) {
                    payloadLen = dis.readUnsignedShort();
                } else if (payloadLen == 127) {
                    payloadLen = dis.readLong();
                }

                byte[] maskKey = null;
                if (masked) {
                    maskKey = new byte[4];
                    dis.readFully(maskKey);
                }

                if (payloadLen < 0 || payloadLen > 2 * 1024 * 1024) {
                    throw new IOException("Invalid frame length: " + payloadLen);
                }

                byte[] payload = new byte[(int) payloadLen];
                dis.readFully(payload);

                if (masked && maskKey != null) {
                    for (int i = 0; i < payload.length; i++) {
                        payload[i] = (byte) (payload[i] ^ maskKey[i % 4]);
                    }
                }

                if (opcode == 0x1) { // Text frame
                    String msg = new String(payload, StandardCharsets.UTF_8);
                    listener.onMessage(msg);
                } else if (opcode == 0x8) { // Close frame
                    break;
                } else if (opcode == 0x9) { // Ping -> Pong
                    sendFrame(0xA, payload);
                }
            }
            listener.onDisconnected("Closed");
        } catch (Exception e) {
            if (running) {
                listener.onDisconnected(e.getMessage() != null ? e.getMessage() : "Error");
            }
        } finally {
            close();
        }
    }

    private void sendFrame(int opcode, byte[] payload) throws IOException {
        synchronized (writeLock) {
            if (outStream == null) return;
            ByteArrayOutputStream frame = new ByteArrayOutputStream();
            frame.write(0x80 | (opcode & 0x0F));
            int len = payload.length;
            if (len <= 125) {
                frame.write(0x80 | len);
            } else if (len <= 65535) {
                frame.write(0x80 | 126);
                frame.write((len >> 8) & 0xFF);
                frame.write(len & 0xFF);
            } else {
                frame.write(0x80 | 127);
                for (int shift = 56; shift >= 0; shift -= 8) {
                    frame.write((int) (((long) len >> shift) & 0xFF));
                }
            }
            byte[] mask = new byte[4];
            random.nextBytes(mask);
            frame.write(mask);
            for (int i = 0; i < len; i++) {
                frame.write(payload[i] ^ mask[i % 4]);
            }
            outStream.write(frame.toByteArray());
            outStream.flush();
        }
    }

    private String readAsciiLine(InputStream in) throws IOException {
        ByteArrayOutputStream buf = new ByteArrayOutputStream();
        int prev = -1;
        while (true) {
            int cur = in.read();
            if (cur == -1) {
                return buf.size() == 0 ? null : buf.toString("US-ASCII");
            }
            if (prev == '\r' && cur == '\n') {
                byte[] bytes = buf.toByteArray();
                return new String(bytes, 0, Math.max(0, bytes.length - 1), StandardCharsets.US_ASCII);
            }
            buf.write(cur);
            prev = cur;
        }
    }
}
