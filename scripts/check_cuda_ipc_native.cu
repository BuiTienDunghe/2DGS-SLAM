// Small, dependency-free CUDA IPC diagnostics.
//
// Each invocation is a producer/consumer pair.  The producer keeps its
// allocation/event alive until the consumer has closed its imported handle.
// This is intentionally separate from PyTorch so that failures can be
// attributed to a CUDA runtime API rather than torch.multiprocessing.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <poll.h>
#include <signal.h>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace {

constexpr uint32_t kWireMagic = 0x43495031;  // "CIP1"
constexpr uint32_t kWireVersion = 1;
constexpr uint32_t kKindMemory = 1;
constexpr uint32_t kKindEvent = 2;
constexpr uint32_t kOrderBefore = 1;
constexpr uint32_t kOrderAfter = 2;
constexpr uint32_t kCommandGo = 0x47;  // G
constexpr size_t kApiBytes = 64;

enum AckStage : int32_t {
  kAckOpen = 1,
  kAckPass = 2,
  kAckClosed = 3,
  kAckFail = 4,
};

struct WireHeader {
  uint32_t magic;
  uint32_t version;
  uint32_t kind;
  uint32_t ordering;
  uint64_t payload_bytes;
  uint64_t pattern_seed;
  unsigned char handle[64];
};

struct Ack {
  uint32_t magic;
  int32_t stage;
  int32_t cuda_code;
  uint64_t checksum;
  char failing_api[kApiBytes];
  char detail[256];
};

static_assert(sizeof(WireHeader) == 96, "wire header changed");
static_assert(sizeof(Ack) == 344, "ack changed");

struct CaseResult {
  std::string kind;
  std::string ordering;
  uint64_t payload_bytes = 0;
  std::string status = "FAIL";
  std::string failing_api;
  int cuda_code = 0;
  std::string cuda_name;
  std::string cuda_message;
  int child_exit = -1;
  bool child_opened = false;
  bool child_opened_checked = false;
  bool child_values_exact = false;
  bool child_values_exact_checked = false;
  bool sender_values_exact_before = false;
  bool sender_values_exact_before_checked = false;
  bool sender_values_unchanged = false;
  bool sender_values_unchanged_checked = false;
  bool child_closed = false;
  bool child_closed_checked = false;
  bool event_observed = false;
  bool event_observed_checked = false;
  uint64_t expected_checksum = 0;
  uint64_t child_checksum = 0;
  uint64_t sender_checksum_before = 0;
  uint64_t sender_checksum_after = 0;
  bool child_checksum_checked = false;
  bool sender_checksum_before_checked = false;
  bool sender_checksum_after_checked = false;
  bool timed_out = false;
  std::string detail;
};

std::string json_escape(const std::string &input) {
  std::string out;
  out.reserve(input.size() + 8);
  for (unsigned char c : input) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out += static_cast<char>(c);
        }
    }
  }
  return out;
}

void set_error(CaseResult *result, const char *api, cudaError_t error) {
  result->failing_api = api;
  result->cuda_code = static_cast<int>(error);
  result->cuda_name = cudaGetErrorName(error);
  result->cuda_message = cudaGetErrorString(error);
  result->detail = std::string(api) + " failed: " + result->cuda_name +
                   " (" + std::to_string(result->cuda_code) + "): " +
                   result->cuda_message;
}

void set_non_cuda_error(CaseResult *result, const std::string &detail) {
  result->detail = detail;
}

bool write_full(int fd, const void *buffer, size_t size) {
  const auto *bytes = static_cast<const unsigned char *>(buffer);
  size_t done = 0;
  while (done < size) {
    ssize_t n = write(fd, bytes + done, size - done);
    if (n > 0) {
      done += static_cast<size_t>(n);
      continue;
    }
    if (n < 0 && errno == EINTR) continue;
    return false;
  }
  return true;
}

bool read_full(int fd, void *buffer, size_t size) {
  auto *bytes = static_cast<unsigned char *>(buffer);
  size_t done = 0;
  while (done < size) {
    ssize_t n = read(fd, bytes + done, size - done);
    if (n > 0) {
      done += static_cast<size_t>(n);
      continue;
    }
    if (n < 0 && errno == EINTR) continue;
    return false;
  }
  return true;
}

bool send_ack(int fd, AckStage stage, cudaError_t error, const char *failing_api,
              const std::string &detail, uint64_t checksum = 0) {
  Ack ack{};
  ack.magic = kWireMagic;
  ack.stage = stage;
  ack.cuda_code = static_cast<int32_t>(error);
  ack.checksum = checksum;
  std::snprintf(ack.failing_api, sizeof(ack.failing_api), "%s", failing_api);
  std::snprintf(ack.detail, sizeof(ack.detail), "%s", detail.c_str());
  return write_full(fd, &ack, sizeof(ack));
}

bool send_ack(int fd, AckStage stage, cudaError_t error, const std::string &detail) {
  return send_ack(fd, stage, error, "", detail);
}

bool send_ok(int fd, AckStage stage, const char *detail, uint64_t checksum = 0) {
  return send_ack(fd, stage, cudaSuccess, "", detail, checksum);
}

bool send_failure(int fd, const char *api, cudaError_t error) {
  const char *name = cudaGetErrorName(error);
  const char *message = cudaGetErrorString(error);
  std::string detail = std::string(api) + " failed: " + name + " (" +
                       std::to_string(static_cast<int>(error)) + "): " + message;
  return send_ack(fd, kAckFail, error, api, detail);
}

uint32_t pattern_value(uint64_t seed, size_t index) {
  uint32_t x = static_cast<uint32_t>(seed) ^
               static_cast<uint32_t>(index * 0x45d9f3bu);
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

std::vector<uint32_t> make_pattern(uint64_t seed, size_t count) {
  std::vector<uint32_t> values(count);
  for (size_t i = 0; i < count; ++i) values[i] = pattern_value(seed, i);
  return values;
}

uint64_t checksum_values(const std::vector<uint32_t> &values) {
  constexpr uint64_t kOffset = 1469598103934665603ull;
  constexpr uint64_t kPrime = 1099511628211ull;
  uint64_t checksum = kOffset;
  for (uint32_t value : values) {
    for (unsigned shift = 0; shift < 32; shift += 8) {
      checksum ^= static_cast<unsigned char>(value >> shift);
      checksum *= kPrime;
    }
  }
  return checksum;
}

bool wait_for_byte(int fd, unsigned char expected, int timeout_seconds) {
  struct pollfd pfd{fd, POLLIN, 0};
  int timeout_ms = timeout_seconds > 0 ? timeout_seconds * 1000 : 1000;
  int ready;
  do {
    ready = poll(&pfd, 1, timeout_ms);
  } while (ready < 0 && errno == EINTR);
  if (ready <= 0 || !(pfd.revents & POLLIN)) return false;
  unsigned char value = 0;
  return read_full(fd, &value, sizeof(value)) && value == expected;
}

int child_memory(int control_fd, int ack_fd, int device) {
  WireHeader header{};
  if (!read_full(control_fd, &header, sizeof(header))) return 21;
  if (header.magic != kWireMagic || header.version != kWireVersion ||
      header.kind != kKindMemory || header.payload_bytes == 0 ||
      header.payload_bytes % sizeof(uint32_t) != 0) {
    send_ack(ack_fd, kAckFail, cudaErrorInvalidValue, "wire header validation",
             "invalid memory wire header");
    return 22;
  }

  cudaError_t error = cudaSetDevice(device);
  if (error != cudaSuccess) {
    send_failure(ack_fd, "cudaSetDevice(child)", error);
    return 23;
  }
  void *mapped = nullptr;
  cudaIpcMemHandle_t handle{};
  std::memcpy(&handle, header.handle, sizeof(handle));
  error = cudaIpcOpenMemHandle(&mapped, handle, cudaIpcMemLazyEnablePeerAccess);
  if (error != cudaSuccess) {
    send_failure(ack_fd, "cudaIpcOpenMemHandle", error);
    return 24;
  }
  if (!send_ok(ack_fd, kAckOpen, "cudaIpcOpenMemHandle succeeded")) {
    cudaIpcCloseMemHandle(mapped);
    return 25;
  }

  if (header.ordering == kOrderBefore &&
      !wait_for_byte(control_fd, kCommandGo, 30)) {
    send_ack(ack_fd, kAckFail, cudaErrorTimeout,
             "handle-before-write: timed out waiting for producer write");
    cudaIpcCloseMemHandle(mapped);
    return 26;
  }

  const size_t count = static_cast<size_t>(header.payload_bytes / sizeof(uint32_t));
  std::vector<uint32_t> received(count);
  error = cudaMemcpy(received.data(), mapped, header.payload_bytes,
                     cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) {
    send_failure(ack_fd, "cudaMemcpy(child DtoH)", error);
    cudaIpcCloseMemHandle(mapped);
    return 27;
  }
  const auto expected = make_pattern(header.pattern_seed, count);
  const uint64_t expected_checksum = checksum_values(expected);
  const uint64_t received_checksum = checksum_values(received);
  if (received != expected || received_checksum != expected_checksum) {
    send_ack(ack_fd, kAckFail, cudaErrorUnknown, "child checksum validation",
             "child received values are not byte-exact", received_checksum);
    cudaIpcCloseMemHandle(mapped);
    return 28;
  }
  if (!send_ok(ack_fd, kAckPass, "child received byte-exact values",
               received_checksum)) {
    cudaIpcCloseMemHandle(mapped);
    return 29;
  }
  error = cudaIpcCloseMemHandle(mapped);
  if (error != cudaSuccess) {
    send_failure(ack_fd, "cudaIpcCloseMemHandle", error);
    return 30;
  }
  if (!send_ok(ack_fd, kAckClosed, "cudaIpcCloseMemHandle succeeded")) return 31;
  return 0;
}

int child_event(int control_fd, int ack_fd, int device) {
  WireHeader header{};
  if (!read_full(control_fd, &header, sizeof(header))) return 41;
  if (header.magic != kWireMagic || header.version != kWireVersion ||
      header.kind != kKindEvent) {
    send_ack(ack_fd, kAckFail, cudaErrorInvalidValue, "wire header validation",
             "invalid event wire header");
    return 42;
  }
  cudaError_t error = cudaSetDevice(device);
  if (error != cudaSuccess) {
    send_failure(ack_fd, "cudaSetDevice(child)", error);
    return 43;
  }
  cudaIpcEventHandle_t handle{};
  std::memcpy(&handle, header.handle, sizeof(handle));
  cudaEvent_t event = nullptr;
  error = cudaIpcOpenEventHandle(&event, handle);
  if (error != cudaSuccess) {
    send_failure(ack_fd, "cudaIpcOpenEventHandle", error);
    return 44;
  }
  if (!send_ok(ack_fd, kAckOpen, "cudaIpcOpenEventHandle succeeded")) {
    cudaEventDestroy(event);
    return 45;
  }
  error = cudaEventSynchronize(event);
  if (error != cudaSuccess) {
    send_failure(ack_fd, "cudaEventSynchronize(child)", error);
    cudaEventDestroy(event);
    return 46;
  }
  if (!send_ok(ack_fd, kAckPass, "child observed producer event completion")) {
    cudaEventDestroy(event);
    return 47;
  }
  error = cudaEventDestroy(event);
  if (error != cudaSuccess) {
    send_failure(ack_fd, "cudaEventDestroy(child)", error);
    return 48;
  }
  if (!send_ok(ack_fd, kAckClosed, "cudaEventDestroy succeeded")) return 49;
  return 0;
}

struct AckWait {
  bool received = false;
  Ack ack{};
};

void mark_failure_checked(CaseResult *result, const char *api) {
  if (std::strcmp(api, "cudaIpcOpenMemHandle") == 0) {
    result->child_opened_checked = true;
  } else if (std::strcmp(api, "cudaIpcOpenEventHandle") == 0) {
    result->child_opened_checked = true;
  } else if (std::strcmp(api, "cudaMemcpy(child DtoH)") == 0 ||
             std::strcmp(api, "child checksum validation") == 0) {
    result->child_values_exact_checked = true;
    result->child_checksum_checked = true;
  } else if (std::strcmp(api, "cudaIpcCloseMemHandle") == 0 ||
             std::strcmp(api, "cudaEventDestroy(child)") == 0) {
    result->child_closed_checked = true;
  } else if (std::strcmp(api, "cudaEventSynchronize(child)") == 0) {
    result->event_observed_checked = true;
  }
}

AckWait wait_for_ack(int ack_fd, pid_t child, AckStage wanted, int timeout_seconds,
                     CaseResult *result) {
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::seconds(timeout_seconds);
  while (std::chrono::steady_clock::now() < deadline) {
    struct pollfd pfd{ack_fd, POLLIN, 0};
    auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
        deadline - std::chrono::steady_clock::now());
    int poll_ms = static_cast<int>(std::clamp<long long>(remaining.count(), 1, 100));
    int ready;
    do {
      ready = poll(&pfd, 1, poll_ms);
    } while (ready < 0 && errno == EINTR);
    if (ready < 0) {
      set_non_cuda_error(result, std::string("poll failed: ") + std::strerror(errno));
      return {};
    }
    if (ready == 0) continue;
    if (pfd.revents & (POLLERR | POLLNVAL)) {
      set_non_cuda_error(result, "ack pipe reported POLLERR/POLLNVAL");
      return {};
    }
    if (pfd.revents & (POLLIN | POLLHUP)) {
      Ack ack{};
      if (!read_full(ack_fd, &ack, sizeof(ack))) {
        set_non_cuda_error(result, "ack pipe closed before a complete acknowledgement");
        return {};
      }
      if (ack.magic != kWireMagic) {
        set_non_cuda_error(result, "ack magic mismatch");
        return {};
      }
      if (ack.stage == kAckFail) {
        result->failing_api = ack.failing_api[0] == '\0' ? "child-reported"
                                                            : ack.failing_api;
        result->cuda_code = ack.cuda_code;
        result->cuda_name = ack.cuda_code == 0 ? "unknown" : cudaGetErrorName(
            static_cast<cudaError_t>(ack.cuda_code));
        result->cuda_message = ack.detail;
        result->detail = ack.detail;
        if (result->kind == "memory" &&
            std::strcmp(result->failing_api.c_str(), "child checksum validation") == 0) {
          result->child_checksum = ack.checksum;
          result->child_checksum_checked = true;
        }
        mark_failure_checked(result, result->failing_api.c_str());
        return {true, ack};
      }
      if (ack.stage == kAckOpen) {
        result->child_opened = true;
        result->child_opened_checked = true;
      }
      if (ack.stage == kAckPass) {
        if (result->kind == "memory") {
          result->child_values_exact = true;
          result->child_values_exact_checked = true;
          result->child_checksum = ack.checksum;
          result->child_checksum_checked = true;
        } else {
          result->event_observed = true;
          result->event_observed_checked = true;
        }
      }
      if (ack.stage == kAckClosed) {
        result->child_closed = true;
        result->child_closed_checked = true;
      }
      if (ack.stage == wanted) return {true, ack};
    }
    int child_status = 0;
    pid_t waited = waitpid(child, &child_status, WNOHANG);
    if (waited == child) {
      result->child_exit = WIFEXITED(child_status) ? WEXITSTATUS(child_status)
                                                    : -WTERMSIG(child_status);
      set_non_cuda_error(result, "child exited before the expected CUDA IPC acknowledgement");
      return {};
    }
    if (waited < 0 && errno != EINTR) {
      set_non_cuda_error(result, std::string("waitpid failed: ") + std::strerror(errno));
      return {};
    }
  }
  result->timed_out = true;
  set_non_cuda_error(result, "timed out waiting for child CUDA IPC acknowledgement");
  return {};
}

void terminate_child(pid_t child, CaseResult *result) {
  if (child <= 0) return;
  int status = 0;
  pid_t waited = waitpid(child, &status, WNOHANG);
  if (waited == 0) {
    // A child that has just reported a CUDA failure normally exits on its
    // own.  Give it a short grace period so the report retains its real exit
    // code; only terminate a genuinely stuck child.
    const auto grace_deadline = std::chrono::steady_clock::now() +
                                std::chrono::seconds(2);
    do {
      waited = waitpid(child, &status, WNOHANG);
      if (waited == child) break;
      usleep(10000);
    } while (std::chrono::steady_clock::now() < grace_deadline);
    if (waited == 0) {
      kill(child, SIGTERM);
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
      do {
        waited = waitpid(child, &status, WNOHANG);
        if (waited == child) break;
        usleep(10000);
      } while (std::chrono::steady_clock::now() < deadline);
    }
    if (waited == 0) {
      kill(child, SIGKILL);
      waitpid(child, &status, 0);
      waited = child;
    }
  }
  if (waited == child) {
    result->child_exit = WIFEXITED(status) ? WEXITSTATUS(status) : -WTERMSIG(status);
  }
}

bool make_child(const std::string &self, uint32_t kind, uint32_t ordering,
                int device, int control_read, int control_write, int ack_read,
                int ack_write, pid_t *child) {
  *child = fork();
  if (*child < 0) return false;
  if (*child == 0) {
    close(control_write);
    close(ack_read);
    std::string kind_arg = kind == kKindMemory ? "memory" : "event";
    std::string order_arg = ordering == kOrderBefore ? "before" : "after";
    std::string device_arg = std::to_string(device);
    std::string control_arg = std::to_string(control_read);
    std::string ack_arg = std::to_string(ack_write);
    char *const args[] = {const_cast<char *>(self.c_str()), const_cast<char *>("--child"),
                          const_cast<char *>(kind_arg.c_str()),
                          const_cast<char *>(order_arg.c_str()),
                          const_cast<char *>(device_arg.c_str()),
                          const_cast<char *>(control_arg.c_str()),
                          const_cast<char *>(ack_arg.c_str()), nullptr};
    execv(self.c_str(), args);
    _exit(127);
  }
  return true;
}

std::string ordering_name(uint32_t kind, uint32_t ordering) {
  if (kind == kKindMemory)
    return ordering == kOrderBefore ? "handle_before_write" : "write_before_handle";
  return ordering == kOrderBefore ? "handle_before_record" : "record_before_handle";
}

CaseResult run_memory(const std::string &self, uint32_t ordering, int device,
                      uint64_t payload_bytes, int timeout_seconds) {
  CaseResult result;
  result.kind = "memory";
  result.ordering = ordering_name(kKindMemory, ordering);
  result.payload_bytes = payload_bytes;
  if (payload_bytes == 0 || payload_bytes % sizeof(uint32_t) != 0) {
    set_non_cuda_error(&result, "payload bytes must be a nonzero multiple of 4");
    return result;
  }
  const size_t count = static_cast<size_t>(payload_bytes / sizeof(uint32_t));
  constexpr uint64_t kSeed = 0x13579bdf2468ace0ull;
  const auto expected = make_pattern(kSeed, count);
  result.expected_checksum = checksum_values(expected);
  void *allocation = nullptr;
  cudaError_t error = cudaMalloc(&allocation, payload_bytes);
  if (error != cudaSuccess) {
    set_error(&result, "cudaMalloc", error);
    return result;
  }
  std::vector<uint32_t> sender_before(count);
  if (ordering == kOrderAfter) {
    error = cudaMemcpy(allocation, expected.data(), payload_bytes, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
      set_error(&result, "cudaMemcpy(sender HtoD)", error);
      cudaFree(allocation);
      return result;
    }
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
      set_error(&result, "cudaDeviceSynchronize(sender write)", error);
      cudaFree(allocation);
      return result;
    }
  }
  if (ordering == kOrderAfter) {
    error = cudaMemcpy(sender_before.data(), allocation, payload_bytes,
                       cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
      set_error(&result, "cudaMemcpy(sender preflight DtoH)", error);
      cudaFree(allocation);
      return result;
    }
    result.sender_values_exact_before = sender_before == expected;
    result.sender_values_exact_before_checked = true;
    result.sender_checksum_before = checksum_values(sender_before);
    result.sender_checksum_before_checked = true;
  }

  cudaIpcMemHandle_t handle{};
  error = cudaIpcGetMemHandle(&handle, allocation);
  if (error != cudaSuccess) {
    set_error(&result, "cudaIpcGetMemHandle", error);
    cudaFree(allocation);
    return result;
  }
  int to_child[2] = {-1, -1};
  int from_child[2] = {-1, -1};
  if (pipe(to_child) != 0 || pipe(from_child) != 0) {
    set_non_cuda_error(&result, std::string("pipe failed: ") + std::strerror(errno));
    if (to_child[0] >= 0) close(to_child[0]);
    if (to_child[1] >= 0) close(to_child[1]);
    if (from_child[0] >= 0) close(from_child[0]);
    if (from_child[1] >= 0) close(from_child[1]);
    cudaFree(allocation);
    return result;
  }
  pid_t child = -1;
  if (!make_child(self, kKindMemory, ordering, device, to_child[0], to_child[1],
                  from_child[0], from_child[1], &child)) {
    set_non_cuda_error(&result, std::string("fork failed: ") + std::strerror(errno));
    close(to_child[0]); close(to_child[1]); close(from_child[0]); close(from_child[1]);
    cudaFree(allocation);
    return result;
  }
  close(to_child[0]);
  close(from_child[1]);
  WireHeader header{};
  header.magic = kWireMagic;
  header.version = kWireVersion;
  header.kind = kKindMemory;
  header.ordering = ordering;
  header.payload_bytes = payload_bytes;
  header.pattern_seed = kSeed;
  std::memcpy(header.handle, &handle, sizeof(handle));
  if (!write_full(to_child[1], &header, sizeof(header))) {
    set_non_cuda_error(&result, "failed to send memory IPC handle to child");
    terminate_child(child, &result);
    close(to_child[1]); close(from_child[0]); cudaFree(allocation);
    return result;
  }
  AckWait open = wait_for_ack(from_child[0], child, kAckOpen, timeout_seconds, &result);
  if (!open.received || result.failing_api.size() != 0 || result.timed_out) {
    terminate_child(child, &result);
    close(to_child[1]); close(from_child[0]); cudaFree(allocation);
    return result;
  }
  if (ordering == kOrderBefore) {
    error = cudaMemcpy(allocation, expected.data(), payload_bytes, cudaMemcpyHostToDevice);
    if (error == cudaSuccess) error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
      set_error(&result, "cudaMemcpy/cudaDeviceSynchronize(sender write)", error);
      terminate_child(child, &result);
      close(to_child[1]); close(from_child[0]); cudaFree(allocation);
      return result;
    }
    error = cudaMemcpy(sender_before.data(), allocation, payload_bytes,
                       cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
      set_error(&result, "cudaMemcpy(sender post-write DtoH)", error);
      terminate_child(child, &result);
      close(to_child[1]); close(from_child[0]); cudaFree(allocation);
      return result;
    }
    result.sender_values_exact_before = sender_before == expected;
    result.sender_values_exact_before_checked = true;
    result.sender_checksum_before = checksum_values(sender_before);
    result.sender_checksum_before_checked = true;
    const unsigned char go = kCommandGo;
    if (!write_full(to_child[1], &go, sizeof(go))) {
      set_non_cuda_error(&result, "failed to release child after sender write");
      terminate_child(child, &result);
      close(to_child[1]); close(from_child[0]); cudaFree(allocation);
      return result;
    }
  }
  AckWait closed = wait_for_ack(from_child[0], child, kAckClosed, timeout_seconds, &result);
  if (!closed.received || result.failing_api.size() != 0 || result.timed_out) {
    terminate_child(child, &result);
    close(to_child[1]); close(from_child[0]); cudaFree(allocation);
    return result;
  }
  int child_status = 0;
  if (waitpid(child, &child_status, 0) == child) {
    result.child_exit = WIFEXITED(child_status) ? WEXITSTATUS(child_status)
                                                 : -WTERMSIG(child_status);
  }
  close(to_child[1]);
  close(from_child[0]);
  std::vector<uint32_t> sender_after(count);
  error = cudaMemcpy(sender_after.data(), allocation, payload_bytes, cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) {
    set_error(&result, "cudaMemcpy(sender postflight DtoH)", error);
  } else {
    result.sender_values_unchanged = sender_before == sender_after && sender_after == expected;
    result.sender_values_unchanged_checked = true;
    result.sender_checksum_after = checksum_values(sender_after);
    result.sender_checksum_after_checked = true;
  }
  cudaError_t free_error = cudaFree(allocation);
  if (result.failing_api.empty() && free_error != cudaSuccess)
    set_error(&result, "cudaFree", free_error);
  if (result.failing_api.empty() && result.child_exit == 0 && result.child_closed &&
      result.child_values_exact && result.sender_values_exact_before &&
      result.sender_values_unchanged && result.child_checksum_checked &&
      result.child_checksum == result.expected_checksum &&
      result.sender_checksum_before_checked &&
      result.sender_checksum_before == result.expected_checksum &&
      result.sender_checksum_after_checked &&
      result.sender_checksum_after == result.expected_checksum) {
    result.status = "PASS";
    result.detail = "memory IPC transferred and preserved byte-exact values";
  } else if (result.failing_api.empty()) {
    result.detail = "memory IPC completed without all exact-value invariants";
  }
  return result;
}

CaseResult run_event(const std::string &self, uint32_t ordering, int device,
                     int timeout_seconds) {
  CaseResult result;
  result.kind = "event";
  result.ordering = ordering_name(kKindEvent, ordering);
  cudaEvent_t event = nullptr;
  cudaError_t error = cudaEventCreateWithFlags(&event, cudaEventDisableTiming | cudaEventInterprocess);
  if (error != cudaSuccess) {
    set_error(&result, "cudaEventCreateWithFlags", error);
    return result;
  }
  cudaIpcEventHandle_t handle{};
  error = cudaIpcGetEventHandle(&handle, event);
  if (error != cudaSuccess) {
    set_error(&result, "cudaIpcGetEventHandle", error);
    cudaEventDestroy(event);
    return result;
  }
  if (ordering == kOrderAfter) {
    error = cudaEventRecord(event, nullptr);
    if (error == cudaSuccess) error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
      set_error(&result, "cudaEventRecord/cudaDeviceSynchronize(sender)", error);
      cudaEventDestroy(event);
      return result;
    }
  }
  int to_child[2] = {-1, -1};
  int from_child[2] = {-1, -1};
  if (pipe(to_child) != 0 || pipe(from_child) != 0) {
    set_non_cuda_error(&result, std::string("pipe failed: ") + std::strerror(errno));
    if (to_child[0] >= 0) close(to_child[0]);
    if (to_child[1] >= 0) close(to_child[1]);
    if (from_child[0] >= 0) close(from_child[0]);
    if (from_child[1] >= 0) close(from_child[1]);
    cudaEventDestroy(event);
    return result;
  }
  pid_t child = -1;
  if (!make_child(self, kKindEvent, ordering, device, to_child[0], to_child[1],
                  from_child[0], from_child[1], &child)) {
    set_non_cuda_error(&result, std::string("fork failed: ") + std::strerror(errno));
    close(to_child[0]); close(to_child[1]); close(from_child[0]); close(from_child[1]);
    cudaEventDestroy(event);
    return result;
  }
  close(to_child[0]);
  close(from_child[1]);
  WireHeader header{};
  header.magic = kWireMagic;
  header.version = kWireVersion;
  header.kind = kKindEvent;
  header.ordering = ordering;
  std::memcpy(header.handle, &handle, sizeof(handle));
  if (!write_full(to_child[1], &header, sizeof(header))) {
    set_non_cuda_error(&result, "failed to send event IPC handle to child");
    terminate_child(child, &result);
    close(to_child[1]); close(from_child[0]); cudaEventDestroy(event);
    return result;
  }
  AckWait open = wait_for_ack(from_child[0], child, kAckOpen, timeout_seconds, &result);
  if (!open.received || result.failing_api.size() != 0 || result.timed_out) {
    terminate_child(child, &result);
    close(to_child[1]); close(from_child[0]); cudaEventDestroy(event);
    return result;
  }
  if (ordering == kOrderBefore) {
    error = cudaEventRecord(event, nullptr);
    if (error == cudaSuccess) error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
      set_error(&result, "cudaEventRecord/cudaDeviceSynchronize(sender)", error);
      terminate_child(child, &result);
      close(to_child[1]); close(from_child[0]); cudaEventDestroy(event);
      return result;
    }
  }
  AckWait closed = wait_for_ack(from_child[0], child, kAckClosed, timeout_seconds, &result);
  if (!closed.received || result.failing_api.size() != 0 || result.timed_out) {
    terminate_child(child, &result);
    close(to_child[1]); close(from_child[0]); cudaEventDestroy(event);
    return result;
  }
  int child_status = 0;
  if (waitpid(child, &child_status, 0) == child) {
    result.child_exit = WIFEXITED(child_status) ? WEXITSTATUS(child_status)
                                                 : -WTERMSIG(child_status);
  }
  close(to_child[1]);
  close(from_child[0]);
  error = cudaEventDestroy(event);
  if (result.failing_api.empty() && error != cudaSuccess)
    set_error(&result, "cudaEventDestroy(sender)", error);
  if (result.failing_api.empty() && result.child_exit == 0 && result.child_closed &&
      result.event_observed) {
    result.status = "PASS";
    result.detail = "event IPC observed producer event completion";
  } else if (result.failing_api.empty()) {
    result.detail = "event IPC completed without all invariants";
  }
  return result;
}

void print_result(const CaseResult &result) {
  std::cout << "{\"schema\":\"cuda-ipc-native/v1\","
            << "\"kind\":\"" << json_escape(result.kind) << "\","
            << "\"ordering\":\"" << json_escape(result.ordering) << "\","
            << "\"payload_bytes\":" << result.payload_bytes << ","
            << "\"status\":\"" << json_escape(result.status) << "\","
            << "\"failing_api\":\"" << json_escape(result.failing_api) << "\","
            << "\"cuda_code\":" << result.cuda_code << ","
            << "\"cuda_name\":\"" << json_escape(result.cuda_name) << "\","
            << "\"cuda_message\":\"" << json_escape(result.cuda_message) << "\","
            << "\"child_exit\":" << result.child_exit << ","
            << "\"child_opened\":" << (result.child_opened ? "true" : "false") << ","
            << "\"child_opened_checked\":"
            << (result.child_opened_checked ? "true" : "false") << ","
            << "\"child_values_exact\":" << (result.child_values_exact ? "true" : "false") << ","
            << "\"child_values_exact_checked\":"
            << (result.child_values_exact_checked ? "true" : "false") << ","
            << "\"sender_values_exact_before\":"
            << (result.sender_values_exact_before ? "true" : "false") << ","
            << "\"sender_values_exact_before_checked\":"
            << (result.sender_values_exact_before_checked ? "true" : "false") << ","
            << "\"sender_values_unchanged\":"
            << (result.sender_values_unchanged ? "true" : "false") << ","
            << "\"sender_values_unchanged_checked\":"
            << (result.sender_values_unchanged_checked ? "true" : "false") << ","
            << "\"child_closed\":" << (result.child_closed ? "true" : "false") << ","
            << "\"child_closed_checked\":"
            << (result.child_closed_checked ? "true" : "false") << ","
            << "\"event_observed\":" << (result.event_observed ? "true" : "false") << ","
            << "\"event_observed_checked\":"
            << (result.event_observed_checked ? "true" : "false") << ","
            << "\"expected_checksum\":" << result.expected_checksum << ","
            << "\"child_checksum\":" << result.child_checksum << ","
            << "\"child_checksum_checked\":"
            << (result.child_checksum_checked ? "true" : "false") << ","
            << "\"sender_checksum_before\":" << result.sender_checksum_before << ","
            << "\"sender_checksum_before_checked\":"
            << (result.sender_checksum_before_checked ? "true" : "false") << ","
            << "\"sender_checksum_after\":" << result.sender_checksum_after << ","
            << "\"sender_checksum_after_checked\":"
            << (result.sender_checksum_after_checked ? "true" : "false") << ","
            << "\"timed_out\":" << (result.timed_out ? "true" : "false") << ","
            << "\"detail\":\"" << json_escape(result.detail) << "\"}\n";
  std::cout.flush();
  if (result.status == "PASS") {
    std::cerr << "[native] PASS " << result.kind << '/' << result.ordering << '\n';
  } else {
    std::cerr << "[native] FAIL " << result.kind << '/' << result.ordering << ": "
              << result.detail << '\n';
  }
}

void usage(const char *program) {
  std::cerr << "usage: " << program
            << " --kind memory|event --ordering before|after [--bytes N]"
               " [--timeout-seconds N]\n";
}

int parse_positive(const char *text, int fallback) {
  char *end = nullptr;
  errno = 0;
  long value = std::strtol(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || value <= 0 || value > 3600)
    return fallback;
  return static_cast<int>(value);
}

int parse_nonnegative(const char *text, int fallback) {
  char *end = nullptr;
  errno = 0;
  long value = std::strtol(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || value < 0 || value > 3600)
    return fallback;
  return static_cast<int>(value);
}

bool parse_bytes(const char *text, uint64_t *value) {
  char *end = nullptr;
  errno = 0;
  unsigned long long parsed = std::strtoull(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0') return false;
  *value = static_cast<uint64_t>(parsed);
  return true;
}

int child_entry(int argc, char **argv) {
  if (argc != 7) return 20;
  const uint32_t kind = std::strcmp(argv[2], "memory") == 0 ? kKindMemory : kKindEvent;
  const uint32_t ordering = std::strcmp(argv[3], "before") == 0 ? kOrderBefore : kOrderAfter;
  const int device = parse_positive(argv[4], 0);
  const int control_fd = parse_positive(argv[5], -1);
  const int ack_fd = parse_positive(argv[6], -1);
  if (control_fd < 0 || ack_fd < 0) return 20;
  if (kind == kKindMemory) return child_memory(control_fd, ack_fd, device);
  return child_event(control_fd, ack_fd, device);
}

}  // namespace

int main(int argc, char **argv) {
  if (argc >= 2 && std::strcmp(argv[1], "--child") == 0) return child_entry(argc, argv);
  std::string kind;
  std::string ordering;
  uint64_t bytes = 32;
  int timeout_seconds = 30;
  int device = 0;
  bool argument_error = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--kind") == 0 && i + 1 < argc) kind = argv[++i];
    else if (std::strcmp(argv[i], "--ordering") == 0 && i + 1 < argc) ordering = argv[++i];
    else if (std::strcmp(argv[i], "--bytes") == 0 && i + 1 < argc) {
      argument_error = !parse_bytes(argv[++i], &bytes);
    }
    else if (std::strcmp(argv[i], "--timeout-seconds") == 0 && i + 1 < argc)
      timeout_seconds = parse_positive(argv[++i], -1);
    else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc)
      device = parse_nonnegative(argv[++i], -1);
    else {
      usage(argv[0]);
      return 2;
    }
  }
  if (argument_error || (kind != "memory" && kind != "event") ||
      (ordering != "before" && ordering != "after") || timeout_seconds <= 0 ||
      device < 0 || (kind == "memory" &&
                      (bytes == 0 || bytes % sizeof(uint32_t) != 0)) ||
      bytes > std::numeric_limits<size_t>::max()) {
    usage(argv[0]);
    return 2;
  }
  int count = 0;
  cudaError_t error = cudaGetDeviceCount(&count);
  if (error != cudaSuccess || count <= device) {
    std::cerr << "[native] CUDA unavailable: " << cudaGetErrorName(error) << " ("
              << static_cast<int>(error) << "): " << cudaGetErrorString(error) << '\n';
    return 2;
  }
  error = cudaSetDevice(device);
  if (error != cudaSuccess) {
    std::cerr << "[native] cudaSetDevice failed: " << cudaGetErrorName(error) << " ("
              << static_cast<int>(error) << "): " << cudaGetErrorString(error) << '\n';
    return 2;
  }
  const uint32_t order_code = ordering == "before" ? kOrderBefore : kOrderAfter;
  const std::string self = std::filesystem::absolute(argv[0]).string();
  CaseResult result = kind == "memory"
                          ? run_memory(self, order_code, device, bytes, timeout_seconds)
                          : run_event(self, order_code, device, timeout_seconds);
  print_result(result);
  return result.status == "PASS" ? 0 : (result.timed_out ? 3 : 1);
}
