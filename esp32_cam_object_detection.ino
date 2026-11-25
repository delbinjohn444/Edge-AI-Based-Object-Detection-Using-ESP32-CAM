/*
 * Edge AI Object Detection with ESP32-CAM
 * Target: AI-Thinker ESP32-CAM / OV2640 / PSRAM
 * Framework: Arduino ESP32 + Edge Impulse FOMO
 *
 * Install the exported Edge Impulse Arduino model library before building.
 * All camera setup and image preprocessing code is included in this sketch.
 */
#include <Arduino.h>
#include <esp_heap_caps.h>
#include "esp_camera.h"
#include "img_converters.h"

// Configuration, board pin mapping and image preprocessing.

// 1 = detection (requires your exported Edge Impulse Arduino library).
// 0 = camera-only smoke test (no ML library needed).
#define EDGE_AI_ENABLED 1

constexpr unsigned long SERIAL_BAUD = 115200;
constexpr float MIN_CONFIDENCE = 0.60f;  // Detection threshold.
constexpr unsigned long LOOP_DELAY_MS = 100;
constexpr int CAPTURE_WIDTH = 320;
constexpr int CAPTURE_HEIGHT = 240;
// The common esp32-camera JPEG decoder returns BGR byte order. Check
// camera-only output with a red/blue target; set false if yours returns RGB.
constexpr bool JPEG_OUTPUT_IS_BGR = true;


// AI-Thinker ESP32-CAM (OV2640). Other variants use different GPIOs.
// Reference: Espressif Arduino CameraWebServer camera_pins.h.
inline camera_config_t makeCameraConfig() {
    camera_config_t c{};
    c.pin_pwdn = 32;
    c.pin_reset = -1;
    c.pin_xclk = 0;
    c.pin_sccb_sda = 26;
    c.pin_sccb_scl = 27;
    c.pin_d0 = 5;
    c.pin_d1 = 18;
    c.pin_d2 = 19;
    c.pin_d3 = 21;
    c.pin_d4 = 36;
    c.pin_d5 = 39;
    c.pin_d6 = 34;
    c.pin_d7 = 35;
    c.pin_vsync = 25;
    c.pin_href = 23;
    c.pin_pclk = 22;
    c.xclk_freq_hz = 20000000;
    c.ledc_timer = LEDC_TIMER_0;
    c.ledc_channel = LEDC_CHANNEL_0;
    c.pixel_format = PIXFORMAT_JPEG;
    c.frame_size = FRAMESIZE_QVGA;
    c.jpeg_quality = 12;
    c.fb_count = 1;
    c.fb_location = CAMERA_FB_IN_PSRAM;
    c.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
    return c;
}

#include <stddef.h>
#include <stdint.h>

namespace image_input {

// Center-crop to target aspect ratio, then bilinear resize. Input and output
// buffers must not overlap. Channels are interpolated independently.
inline bool centerCropResize(const uint8_t *src, int sw, int sh,
                             uint8_t *dst, int dw, int dh) {
    if (!src || !dst || src == dst || sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0)
        return false;
    int cw = sw, ch = sh;
    if (static_cast<int64_t>(sw) * dh > static_cast<int64_t>(sh) * dw)
        cw = static_cast<int>(static_cast<int64_t>(sh) * dw / dh);
    else
        ch = static_cast<int>(static_cast<int64_t>(sw) * dh / dw);
    if (cw < 1 || ch < 1) return false;
    const int ox = (sw - cw) / 2, oy = (sh - ch) / 2;
    for (int y = 0; y < dh; ++y) {
        float sy = (y + 0.5f) * ch / dh - 0.5f;
        if (sy < 0) sy = 0;
        if (sy > ch - 1) sy = static_cast<float>(ch - 1);
        const int y0 = static_cast<int>(sy), y1 = y0 + 1 < ch ? y0 + 1 : y0;
        const float fy = sy - y0;
        for (int x = 0; x < dw; ++x) {
            float sx = (x + 0.5f) * cw / dw - 0.5f;
            if (sx < 0) sx = 0;
            if (sx > cw - 1) sx = static_cast<float>(cw - 1);
            const int x0 = static_cast<int>(sx), x1 = x0 + 1 < cw ? x0 + 1 : x0;
            const float fx = sx - x0;
            const size_t p00 = (static_cast<size_t>(oy + y0) * sw + ox + x0) * 3;
            const size_t p01 = (static_cast<size_t>(oy + y0) * sw + ox + x1) * 3;
            const size_t p10 = (static_cast<size_t>(oy + y1) * sw + ox + x0) * 3;
            const size_t p11 = (static_cast<size_t>(oy + y1) * sw + ox + x1) * 3;
            const size_t out = (static_cast<size_t>(y) * dw + x) * 3;
            for (int c = 0; c < 3; ++c) {
                const float a = src[p00+c] * (1-fx) + src[p01+c] * fx;
                const float b = src[p10+c] * (1-fx) + src[p11+c] * fx;
                dst[out+c] = static_cast<uint8_t>(a * (1-fy) + b * fy + 0.5f);
            }
        }
    }
    return true;
}

// Edge Impulse image signals use one numeric 0xRRGGBB value per pixel.
inline uint32_t packedRgb(const uint8_t *p, bool bgr) {
    return (static_cast<uint32_t>(p[bgr ? 2 : 0]) << 16) |
           (static_cast<uint32_t>(p[1]) << 8) | p[bgr ? 0 : 2];
}

inline int readPackedPixels(const uint8_t *pixels, size_t count, bool bgr,
                            size_t offset, size_t length, float *out) {
    if (!pixels || !out || offset > count || length > count - offset) return -1;
    for (size_t i = 0; i < length; ++i)
        out[i] = static_cast<float>(packedRgb(pixels + (offset + i) * 3, bgr));
    return 0;
}

} // namespace image_input


#if EDGE_AI_ENABLED
// Use the header name from the installed Edge Impulse model library.
#include <esp32_cam_fomo_inferencing.h>
#if EI_CLASSIFIER_OBJECT_DETECTION != 1
#error "Export an object detection model (FOMO), not an image classifier."
#endif
#if EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_CAMERA
#error "Export a camera/image impulse."
#endif
static_assert(EI_CLASSIFIER_INPUT_WIDTH > 0 && EI_CLASSIFIER_INPUT_HEIGHT > 0,
              "Invalid model dimensions");
static_assert(EI_CLASSIFIER_INPUT_WIDTH <= CAPTURE_WIDTH &&
              EI_CLASSIFIER_INPUT_HEIGHT <= CAPTURE_HEIGHT,
              "Use a small input size, e.g. 96x96");
constexpr size_t MODEL_PIXELS = static_cast<size_t>(EI_CLASSIFIER_INPUT_WIDTH) *
                                EI_CLASSIFIER_INPUT_HEIGHT;
static uint8_t *modelImage = nullptr;
#endif

constexpr size_t CAPTURE_BYTES = static_cast<size_t>(CAPTURE_WIDTH) * CAPTURE_HEIGHT * 3;
static uint8_t *decodedImage = nullptr;
static unsigned long frameId = 0;

static void haltWithError(const char *message) {
    Serial.print("FATAL: ");
    Serial.println(message);
    while (true) delay(1000);
}

static bool captureImage() {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        Serial.println("ERROR: frame capture failed");
        return false;
    }
    // Decode only the configured QVGA frame, into a dedicated PSRAM buffer.
    const bool valid = fb->width == CAPTURE_WIDTH && fb->height == CAPTURE_HEIGHT &&
                       fb->format == PIXFORMAT_JPEG && fb->len > 0;
    const bool ok = valid && fmt2rgb888(fb->buf, fb->len, fb->format, decodedImage);
    esp_camera_fb_return(fb); // Every acquired frame is returned exactly once.
    if (!ok) Serial.println("ERROR: unexpected frame format or JPEG decode failure");
    return ok;
}

#if EDGE_AI_ENABLED
static int getSignalData(size_t offset, size_t length, float *out) {
    return image_input::readPackedPixels(modelImage, MODEL_PIXELS,
                                         JPEG_OUTPUT_IS_BGR, offset, length, out);
}
#endif

void setup() {
    Serial.begin(SERIAL_BAUD);
    delay(1200);
    Serial.println("\nESP32-CAM | Edge AI object detection");
    if (!psramFound()) haltWithError("PSRAM not detected. Check board selection and PSRAM.");
    camera_config_t camera = makeCameraConfig();
    const esp_err_t err = esp_camera_init(&camera);
    if (err != ESP_OK) {
        Serial.printf("Camera initialization error: 0x%x\n", static_cast<unsigned int>(err));
        haltWithError("Check camera ribbon, GPIO mapping and stable power.");
    }
    decodedImage = static_cast<uint8_t *>(heap_caps_malloc(CAPTURE_BYTES,
                                       MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    if (!decodedImage) haltWithError("Cannot allocate camera decode buffer in PSRAM.");
#if EDGE_AI_ENABLED
    modelImage = static_cast<uint8_t *>(heap_caps_malloc(MODEL_PIXELS * 3,
                                    MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    if (!modelImage) haltWithError("Cannot allocate model input buffer in PSRAM.");
    Serial.printf("Mode: detection | input=%dx%d | threshold=%.2f\n",
                  EI_CLASSIFIER_INPUT_WIDTH, EI_CLASSIFIER_INPUT_HEIGHT, MIN_CONFIDENCE);
#else
    Serial.println("Mode: camera-only smoke test (no ML inference)");
#endif
    delay(1000);
}

void loop() {
    const unsigned long started = millis();
    if (!captureImage()) { delay(500); return; }
    ++frameId;
#if EDGE_AI_ENABLED
    if (!image_input::centerCropResize(decodedImage, CAPTURE_WIDTH, CAPTURE_HEIGHT,
            modelImage, EI_CLASSIFIER_INPUT_WIDTH, EI_CLASSIFIER_INPUT_HEIGHT)) {
        Serial.println("ERROR: image resize failed");
        delay(500);
        return;
    }
    ei::signal_t signal{};
    signal.total_length = MODEL_PIXELS;
    signal.get_data = getSignalData;
    ei_impulse_result_t result{};
    EI_IMPULSE_ERROR status = run_classifier(&signal, &result, false);
    if (status != EI_IMPULSE_OK) {
        Serial.printf("ERROR: inference failed (%d)\n", static_cast<int>(status));
        delay(500);
        return;
    }
    const unsigned long pipelineMs = millis() - started;
    unsigned int count = 0;
    for (size_t i = 0; i < result.bounding_boxes_count; ++i) {
        const auto &box = result.bounding_boxes[i];
        if (box.value <= 0.0f || box.value < MIN_CONFIDENCE) continue;
        const float cx = box.x + box.width / 2.0f;
        const float cy = box.y + box.height / 2.0f;
        Serial.printf("DETECTION frame=%lu label=%s confidence=%.3f center=(%.1f,%.1f) region=(%u,%u,%u,%u)\n",
                      frameId, box.label, box.value, cx, cy,
                      static_cast<unsigned int>(box.x), static_cast<unsigned int>(box.y),
                      static_cast<unsigned int>(box.width), static_cast<unsigned int>(box.height));
        ++count;
    }
    if (!count) Serial.printf("NO_DETECTION frame=%lu\n", frameId);
    Serial.printf("FRAME frame=%lu count=%u dsp_ms=%lu inference_ms=%lu pipeline_ms=%lu free_heap=%u free_psram=%u\n",
                  frameId, count, static_cast<unsigned long>(result.timing.dsp),
                  static_cast<unsigned long>(result.timing.classification), pipelineMs,
                  static_cast<unsigned int>(ESP.getFreeHeap()),
                  static_cast<unsigned int>(ESP.getFreePsram()));
#else
    const size_t mid = (static_cast<size_t>(CAPTURE_HEIGHT / 2) * CAPTURE_WIDTH + CAPTURE_WIDTH / 2) * 3;
    uint32_t color = image_input::packedRgb(decodedImage + mid, JPEG_OUTPUT_IS_BGR);
    Serial.printf("CAMERA_OK frame=%lu size=%dx%d center_rgb=(%u,%u,%u) capture_ms=%lu\n",
                  frameId, CAPTURE_WIDTH, CAPTURE_HEIGHT,
                  static_cast<unsigned int>((color >> 16) & 255),
                  static_cast<unsigned int>((color >> 8) & 255),
                  static_cast<unsigned int>(color & 255), millis() - started);
#endif
    delay(LOOP_DELAY_MS);
}
