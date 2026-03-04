#include "llvm-pass/include/LoopTiling.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;
char LoopTilingPass::ID = 0;

static RegisterPass<LoopTilingPass>
    X("loop-tiling", "Perfect-Nest Loop Tiling Pass", false, false);

void LoopTilingPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<LoopInfoWrapperPass>();
  AU.addRequired<ScalarEvolutionWrapperPass>();
}

bool LoopTilingPass::runOnFunction(Function &F) {
  LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  auto &SE = getAnalysis<ScalarEvolutionWrapperPass>().getSE();
  bool Changed = false;

  for (Loop *L : LI) {
    if (L->getSubLoops().size() == 1)
      tilePerfectNest(L, LI, SE), Changed = true;
  }
  return Changed;
}

void LoopTilingPass::tilePerfectNest(Loop *Outer, LoopInfo &LI, ScalarEvolution &SE) {
  Loop *Inner = Outer->getSubLoops()[0];
  PHINode *iInd = Outer->getCanonicalInductionVariable();
  PHINode *jInd = Inner->getCanonicalInductionVariable();
  if (!iInd || !jInd) return;

  // Bounds
  Value *lb = iInd->getIncomingValueForBlock(Outer->getLoopPreheader());
  auto *BackCnt = SE.getBackedgeTakenCount(Outer);
  if (!BackCnt) return;
  Value *ub = SE.getAddExpr(BackCnt, ConstantInt::get(BackCnt->getType(), 1));

  IRBuilder<> B(Outer->getHeader()->getFirstNonPHI());
  Type *IdxTy = lb->getType();

  // Create tile index ii
  PHINode *ii = B.CreatePHI(IdxTy, 2, "ii");
  ii->addIncoming(lb, Outer->getLoopPreheader());

  // Replace uses of iInd inside the tiled region
  SmallVector<Instruction*, 8> ToReplace;
  for (auto &U : iInd->uses()) {
    if (Instruction *Inst = dyn_cast<Instruction>(U.getUser()))
      if (Outer->contains(Inst))
        ToReplace.push_back(Inst);
  }
  for (auto *Inst : ToReplace) {
    B.SetInsertPoint(Inst);
    Value *off = B.CreateAdd(ii, iInd, "i_off");
    Inst->replaceUsesOfWith(iInd, off);
  }

  // ii_next = ii + T
  Value *TVal = ConstantInt::get(IdxTy, T);
  Value *iiNext = B.CreateAdd(ii, TVal, "ii_next");
  Value *cond = B.CreateICmpSLT(iiNext, ub, "tile_cond");
  BasicBlock *ExitBB = Outer->getExitBlock();
  B.CreateCondBr(cond, Outer->getHeader(), ExitBB);
  ii->addIncoming(iiNext, Outer->getHeader());
}
