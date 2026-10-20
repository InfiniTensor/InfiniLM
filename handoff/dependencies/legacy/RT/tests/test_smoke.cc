#include <infini/rt.h>

#include "test_helper.h"

int main() {
  infini::rt::test::TestContext context;

  context.Expect(true, "The test helper should report passing expectations.");

  return context.ExitCode();
}
