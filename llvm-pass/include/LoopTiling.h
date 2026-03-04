#ifndef LLVM_TRANSFORMS_LOOPTILING_H
#define LLVM_TRANSFORMS_LOOPTILING_H

#include "llvm/IR/Function.h"
#include "llvm/Pass.h"

namespace llvm {

class LoopTilingPass : public FunctionPass {
public:
  static char ID;
  LoopTilingPass(int TileSize = 16) : FunctionPass(ID), T(TileSize) {}

  bool runOnFunction(Function &F) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;

private:
  int T;
  void tilePerfectNest(Loop *Outer, LoopInfo &LI, ScalarEvolution &SE);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_LOOPTILING_H
