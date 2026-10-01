// Optional. Provides libstdc++ symbols (GLIBCXX_3.4.29+) that the Triton prebuilt LLVM static
// libs reference but an old system libstdc++ (gcc 10, GLIBCXX 3.4.28) lacks. Only needed when the
// build fails to link with undefined std::__glibcxx_assert_fail /
// std::__throw_bad_array_new_length / std::__exception_ptr::exception_ptr::_M_release.
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <new>
namespace std {
[[noreturn]] void __throw_bad_array_new_length() { throw std::bad_array_new_length(); }
[[noreturn]] void __glibcxx_assert_fail(const char* file, int line, const char* func, const char* cond) noexcept {
  std::fprintf(stderr, "%s:%d: %s: Assertion '%s' failed.\n", file, line, func, cond);
  std::abort();
}
}  // namespace std
void std::__exception_ptr::exception_ptr::_M_release() noexcept {
  std::__exception_ptr::exception_ptr tmp;
  swap(tmp);  // tmp dtor (exported by libstdc++) drops the reference
}
