#include <curl/curl.h>

#include <chrono>
#include <cstring>
#include <thread>

#include "http_client.h"
#include "logging.h"

class CurlClient : public ouster::util::HttpClient {
   public:
    CurlClient(const std::string& base_url_, int timeout_seconds)
        : HttpClient(base_url_) {
        curl_global_init(CURL_GLOBAL_ALL);
        curl_handle = curl_easy_init();
        curl_easy_setopt(curl_handle, CURLOPT_WRITEFUNCTION,
                         &CurlClient::write_memory_callback);
        curl_easy_setopt(curl_handle, CURLOPT_WRITEDATA, this);
        curl_easy_setopt(curl_handle, CURLOPT_TIMEOUT, timeout_seconds);
    }

    virtual ~CurlClient() override {
        curl_easy_cleanup(curl_handle);
        curl_global_cleanup();
    }

   public:
    std::string get(const std::string& url) const override {
        auto full_url = url_combine(base_url, url);
        return execute_get(full_url);
    }

    std::string encode(const std::string& str) const override {
        auto curl_str_deleter = [](char* s) { curl_free(s); };
        auto encoded_str = std::unique_ptr<char, decltype(curl_str_deleter)>(
            curl_easy_escape(curl_handle, str.c_str(), str.length()),
            curl_str_deleter);
        return std::string(encoded_str.get());
    }

   private:
    static std::string url_combine(const std::string& url1,
                                   const std::string& url2) {
        if (!url1.empty() && !url2.empty()) {
            if (url1[url1.length() - 1] == '/' && url2[0] == '/') {
                return url1 + url2.substr(1);
            }
            if (url1[url1.length() - 1] != '/' && url2[0] != '/') {
                return url1 + '/' + url2;
            }
        }

        return url1 + url2;
    }

    std::string execute_get(const std::string& url, int attempts = 3,
                            int retry_delay_ms = 500) const {
        long http_code = 0;
        if (attempts < 1) {
            throw std::invalid_argument(
                "CurlClient::execute_get: invalid number of retries");
        }
        if (retry_delay_ms < 0) {
            throw std::invalid_argument(
                "CurlClient::execute_get: invalid retry delay");
        }
        curl_easy_setopt(curl_handle, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl_handle, CURLOPT_HTTPGET, 1L);
        while (attempts--) {
            buffer.clear();
            auto res = curl_easy_perform(curl_handle);
            if (res == CURLE_SEND_ERROR) {
                // Specific versions of curl does't play well with the sensor
                // http server. When CURLE_SEND_ERROR happens for the first time
                // silently re-attempting the http request resolves the problem.
                res = curl_easy_perform(curl_handle);
            }
            if (res != CURLE_OK) {
                throw std::runtime_error(
                    "CurlClient::execute_get failed for the url: [" + url +
                    "] with the error message: " + curl_easy_strerror(res));
            }
            curl_easy_getinfo(curl_handle, CURLINFO_RESPONSE_CODE, &http_code);
            if (200 <= http_code && http_code < 300) {
                // HTTP 2XX means a successful response
                return buffer;
            }
            if (attempts && 500 <= http_code && http_code < 600) {
                // HTTP 5XX means a server error, so we should re-attempt.
                // log a warning and sleep before re-attempting
               ouster::sensor::logger().warn(
                  "Re-attempting CurlClient::execute_get after failure for url: [{}] with the code: [{}] - and return: {}",
                  url, http_code, buffer);
                   std::this_thread::sleep_for(
                    std::chrono::milliseconds(retry_delay_ms));
            }
        }

        // the request failed after repeated attempts with a response code other
        // than HTTP 2XX
        throw std::runtime_error(
            std::string("CurlClient::execute_get failed for url: [" + url +
                        "] with the code: [" + std::to_string(http_code) +
                        std::string("] - and return: ") + buffer));
    }

    static size_t write_memory_callback(void* contents, size_t element_size,
                                        size_t elements_count,
                                        void* user_pointer) {
        size_t size_increment = element_size * elements_count;
        auto cc = static_cast<CurlClient*>(user_pointer);
        auto size_current = cc->buffer.size();
        cc->buffer.resize(size_current + size_increment);
        memcpy((void*)&cc->buffer[size_current], contents, size_increment);
        size_current += size_increment;
        return size_increment;
    }

   private:
    CURL* curl_handle;
    mutable std::string buffer;
};
