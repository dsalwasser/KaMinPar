#include "kaminpar-common/perf.h"

#include <cassert>
#include <sstream>
#include <stdexcept>

#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

template <std::size_t N> int run_process(const char *file, const char *const (&argv)[N]) {
  assert((N > 0) && (argv[N - 1] == nullptr));
  return execvp(file, const_cast<char *const *>(argv));
}

} // namespace

namespace kaminpar::perf {

static pid_t perf_child_pid = -1;
static int pipe_read_fd;

void start() {
  int pipe_fds[2];
  if (pipe(pipe_fds) == -1) {
    throw std::runtime_error{"Failed to create a pipe for the perf process"};
  }

  const pid_t parent_pid = getpid();
  const pid_t pid = fork();
  if (pid == -1) {
    throw std::runtime_error{"Failed to fork process that starts perf"};
  }

  if (pid == 0) {
    if (close(pipe_fds[0]) == -1) {
      throw std::runtime_error{"Failed to close read end of the pipe"};
    }

    if (dup2(pipe_fds[1], STDERR_FILENO) == -1) {
      throw std::runtime_error{"Failed to redirect dtandard error output to the pipe"};
    }

    if (close(pipe_fds[1]) == -1) {
      throw std::runtime_error{"Failed to close write end of the pipe"};
    }

    if (setsid() == -1) {
      throw std::runtime_error{"Failed to set session id for the perf process"};
    }

    const std::string parent_pid_str = std::to_string(parent_pid);
    const char *const args[] = {
        "perf", "stat", "-d", "-d", "-d", "-p", parent_pid_str.c_str(), nullptr
    };

    run_process("perf", args);
    throw std::runtime_error{"Failed to execute perf process"};
  } else {
    if (close(pipe_fds[1]) == -1) {
      throw std::runtime_error{"Failed to close write end of the pipe"};
    }

    sleep(3);

    pipe_read_fd = pipe_fds[0];
    perf_child_pid = pid;
  }
}

std::string stop() {
  if (perf_child_pid == -1) {
    throw std::runtime_error{"Only the main process can stop the perf process"};
  }

  if (kill(perf_child_pid, SIGINT) == -1) {
    throw std::runtime_error{"Failed to terminate perf process"};
  }

  if (waitpid(perf_child_pid, NULL, 0) == -1) {
    throw std::runtime_error{"Failed to wait for perf process termination"};
  }

  std::stringstream perf_output;

  ssize_t num_bytes;
  char buffer[1024 + 1];
  while ((num_bytes = read(pipe_read_fd, buffer, sizeof(buffer) - 1)) > 0) {
    buffer[num_bytes] = '\0';
    perf_output << buffer;
  }

  if (close(pipe_read_fd)) {
    throw std::runtime_error{"Failed to close read end of the pipe"};
  }

  return perf_output.str();
}

} // namespace kaminpar::perf
