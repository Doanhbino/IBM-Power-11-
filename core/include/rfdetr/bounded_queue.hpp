#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>

namespace rfdetr {

enum class QueuePolicy { block, drop_oldest };

struct QueueStats {
    std::size_t accepted{}, popped{}, dropped{}, high_watermark{}, pending{};
    bool closed{};
};

// One queue per camera. close() drains; cancel() discards and unblocks producers.
template <typename T> class BoundedQueue {
public:
    BoundedQueue(std::size_t capacity, QueuePolicy policy) : capacity_(capacity), policy_(policy) {
        if (!capacity) throw std::invalid_argument("Queue capacity must be positive");
    }
    bool push(T item) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (policy_ == QueuePolicy::block)
            writable_.wait(lock, [&] { return stats_.closed || items_.size() < capacity_; });
        if (stats_.closed) return false;
        if (items_.size() == capacity_) { items_.pop_front(); ++stats_.dropped; }
        items_.push_back(std::move(item));
        ++stats_.accepted;
        if (items_.size() > stats_.high_watermark) stats_.high_watermark = items_.size();
        return true;
    }
    std::optional<T> try_pop() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (items_.empty()) return std::nullopt;
        T item = std::move(items_.front());
        items_.pop_front();
        ++stats_.popped;
        writable_.notify_one();
        return item;
    }
    void close() {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_.closed = true;
        writable_.notify_all();
    }
    void cancel() {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_.closed = true;
        stats_.dropped += items_.size();
        items_.clear();
        writable_.notify_all();
    }
    QueueStats stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto copy = stats_;
        copy.pending = items_.size();
        return copy;
    }
private:
    const std::size_t capacity_;
    const QueuePolicy policy_;
    mutable std::mutex mutex_;
    std::condition_variable writable_;
    std::deque<T> items_;
    QueueStats stats_;
};

}  // namespace rfdetr
