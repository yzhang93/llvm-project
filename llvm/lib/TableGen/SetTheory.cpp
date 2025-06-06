//===- SetTheory.cpp - Generate ordered sets from DAG expressions ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the SetTheory class that computes ordered sets of
// Records from DAG expressions.
//
//===----------------------------------------------------------------------===//

#include "llvm/TableGen/SetTheory.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>

using namespace llvm;

// Define the standard operators.
namespace {

using RecSet = SetTheory::RecSet;
using RecVec = SetTheory::RecVec;

// (add a, b, ...) Evaluate and union all arguments.
struct AddOp : public SetTheory::Operator {
  void apply(SetTheory &ST, const DagInit *Expr, RecSet &Elts,
             ArrayRef<SMLoc> Loc) override {
    ST.evaluate(Expr->arg_begin(), Expr->arg_end(), Elts, Loc);
  }
};

// (sub Add, Sub, ...) Set difference.
struct SubOp : public SetTheory::Operator {
  void apply(SetTheory &ST, const DagInit *Expr, RecSet &Elts,
             ArrayRef<SMLoc> Loc) override {
    if (Expr->arg_size() < 2)
      PrintFatalError(Loc, "Set difference needs at least two arguments: " +
        Expr->getAsString());
    RecSet Add, Sub;
    ST.evaluate(*Expr->arg_begin(), Add, Loc);
    ST.evaluate(Expr->arg_begin() + 1, Expr->arg_end(), Sub, Loc);
    for (const auto &I : Add)
      if (!Sub.count(I))
        Elts.insert(I);
  }
};

// (and S1, S2) Set intersection.
struct AndOp : public SetTheory::Operator {
  void apply(SetTheory &ST, const DagInit *Expr, RecSet &Elts,
             ArrayRef<SMLoc> Loc) override {
    if (Expr->arg_size() != 2)
      PrintFatalError(Loc, "Set intersection requires two arguments: " +
        Expr->getAsString());
    RecSet S1, S2;
    ST.evaluate(Expr->arg_begin()[0], S1, Loc);
    ST.evaluate(Expr->arg_begin()[1], S2, Loc);
    for (const auto &I : S1)
      if (S2.count(I))
        Elts.insert(I);
  }
};

// SetIntBinOp - Abstract base class for (Op S, N) operators.
struct SetIntBinOp : public SetTheory::Operator {
  virtual void apply2(SetTheory &ST, const DagInit *Expr, RecSet &Set,
                      int64_t N, RecSet &Elts, ArrayRef<SMLoc> Loc) = 0;

  void apply(SetTheory &ST, const DagInit *Expr, RecSet &Elts,
             ArrayRef<SMLoc> Loc) override {
    if (Expr->arg_size() != 2)
      PrintFatalError(Loc, "Operator requires (Op Set, Int) arguments: " +
        Expr->getAsString());
    RecSet Set;
    ST.evaluate(Expr->arg_begin()[0], Set, Loc);
    const auto *II = dyn_cast<IntInit>(Expr->arg_begin()[1]);
    if (!II)
      PrintFatalError(Loc, "Second argument must be an integer: " +
        Expr->getAsString());
    apply2(ST, Expr, Set, II->getValue(), Elts, Loc);
  }
};

// (shl S, N) Shift left, remove the first N elements.
struct ShlOp : public SetIntBinOp {
  void apply2(SetTheory &ST, const DagInit *Expr, RecSet &Set, int64_t N,
              RecSet &Elts, ArrayRef<SMLoc> Loc) override {
    if (N < 0)
      PrintFatalError(Loc, "Positive shift required: " +
        Expr->getAsString());
    if (unsigned(N) < Set.size())
      Elts.insert(Set.begin() + N, Set.end());
  }
};

// (trunc S, N) Truncate after the first N elements.
struct TruncOp : public SetIntBinOp {
  void apply2(SetTheory &ST, const DagInit *Expr, RecSet &Set, int64_t N,
              RecSet &Elts, ArrayRef<SMLoc> Loc) override {
    if (N < 0)
      PrintFatalError(Loc, "Positive length required: " +
        Expr->getAsString());
    if (unsigned(N) > Set.size())
      N = Set.size();
    Elts.insert(Set.begin(), Set.begin() + N);
  }
};

// Left/right rotation.
struct RotOp : public SetIntBinOp {
  const bool Reverse;

  RotOp(bool Rev) : Reverse(Rev) {}

  void apply2(SetTheory &ST, const DagInit *Expr, RecSet &Set, int64_t N,
              RecSet &Elts, ArrayRef<SMLoc> Loc) override {
    if (Reverse)
      N = -N;
    // N > 0 -> rotate left, N < 0 -> rotate right.
    if (Set.empty())
      return;
    if (N < 0)
      N = Set.size() - (-N % Set.size());
    else
      N %= Set.size();
    Elts.insert(Set.begin() + N, Set.end());
    Elts.insert(Set.begin(), Set.begin() + N);
  }
};

// (decimate S, N) Pick every N'th element of S.
struct DecimateOp : public SetIntBinOp {
  void apply2(SetTheory &ST, const DagInit *Expr, RecSet &Set, int64_t N,
              RecSet &Elts, ArrayRef<SMLoc> Loc) override {
    if (N <= 0)
      PrintFatalError(Loc, "Positive stride required: " +
        Expr->getAsString());
    for (unsigned I = 0; I < Set.size(); I += N)
      Elts.insert(Set[I]);
  }
};

// (interleave S1, S2, ...) Interleave elements of the arguments.
struct InterleaveOp : public SetTheory::Operator {
  void apply(SetTheory &ST, const DagInit *Expr, RecSet &Elts,
             ArrayRef<SMLoc> Loc) override {
    // Evaluate the arguments individually.
    SmallVector<RecSet, 4> Values(Expr->getNumArgs());
    unsigned MaxSize = 0;
    for (auto [Arg, Value] : zip(Expr->getArgs(), Values)) {
      ST.evaluate(Arg, Value, Loc);
      MaxSize = std::max(MaxSize, unsigned(Value.size()));
    }
    // Interleave arguments into Elts.
    for (unsigned n = 0; n != MaxSize; ++n)
      for (const RecSet &Value : Values)
        if (n < Value.size())
          Elts.insert(Value[n]);
  }
};

// (sequence "Format", From, To) Generate a sequence of records by name.
struct SequenceOp : public SetTheory::Operator {
  void apply(SetTheory &ST, const DagInit *Expr, RecSet &Elts,
             ArrayRef<SMLoc> Loc) override {
    int Step = 1;
    if (Expr->arg_size() > 4)
      PrintFatalError(Loc, "Bad args to (sequence \"Format\", From, To): " +
        Expr->getAsString());
    if (Expr->arg_size() == 4) {
      if (const auto *II = dyn_cast<IntInit>(Expr->arg_begin()[3]))
        Step = II->getValue();
      else
        PrintFatalError(Loc, "Stride must be an integer: " +
          Expr->getAsString());
    }

    std::string Format;
    if (const auto *SI = dyn_cast<StringInit>(Expr->arg_begin()[0]))
      Format = std::string(SI->getValue());
    else
      PrintFatalError(Loc,  "Format must be a string: " + Expr->getAsString());

    int64_t From, To;
    if (const auto *II = dyn_cast<IntInit>(Expr->arg_begin()[1]))
      From = II->getValue();
    else
      PrintFatalError(Loc, "From must be an integer: " + Expr->getAsString());
    if (From < 0 || From >= (1 << 30))
      PrintFatalError(Loc, "From out of range");

    if (const auto *II = dyn_cast<IntInit>(Expr->arg_begin()[2]))
      To = II->getValue();
    else
      PrintFatalError(Loc, "To must be an integer: " + Expr->getAsString());
    if (To < 0 || To >= (1 << 30))
      PrintFatalError(Loc, "To out of range");

    const RecordKeeper &Records =
        cast<DefInit>(Expr->getOperator())->getDef()->getRecords();

    Step *= From <= To ? 1 : -1;
    while (true) {
      if (Step > 0 && From > To)
        break;
      else if (Step < 0 && From < To)
        break;
      std::string Name;
      raw_string_ostream OS(Name);
      OS << format(Format.c_str(), unsigned(From));
      const Record *Rec = Records.getDef(Name);
      if (!Rec)
        PrintFatalError(Loc, "No def named '" + Name + "': " +
          Expr->getAsString());
      // Try to reevaluate Rec in case it is a set.
      if (const RecVec *Result = ST.expand(Rec))
        Elts.insert_range(*Result);
      else
        Elts.insert(Rec);

      From += Step;
    }
  }
};

// Expand a Def into a set by evaluating one of its fields.
struct FieldExpander : public SetTheory::Expander {
  StringRef FieldName;

  FieldExpander(StringRef fn) : FieldName(fn) {}

  void expand(SetTheory &ST, const Record *Def, RecSet &Elts) override {
    ST.evaluate(Def->getValueInit(FieldName), Elts, Def->getLoc());
  }
};

} // end anonymous namespace

// Pin the vtables to this file.
void SetTheory::Operator::anchor() {}
void SetTheory::Expander::anchor() {}

SetTheory::SetTheory() {
  addOperator("add", std::make_unique<AddOp>());
  addOperator("sub", std::make_unique<SubOp>());
  addOperator("and", std::make_unique<AndOp>());
  addOperator("shl", std::make_unique<ShlOp>());
  addOperator("trunc", std::make_unique<TruncOp>());
  addOperator("rotl", std::make_unique<RotOp>(false));
  addOperator("rotr", std::make_unique<RotOp>(true));
  addOperator("decimate", std::make_unique<DecimateOp>());
  addOperator("interleave", std::make_unique<InterleaveOp>());
  addOperator("sequence", std::make_unique<SequenceOp>());
}

void SetTheory::addOperator(StringRef Name, std::unique_ptr<Operator> Op) {
  Operators[Name] = std::move(Op);
}

void SetTheory::addExpander(StringRef ClassName, std::unique_ptr<Expander> E) {
  Expanders[ClassName] = std::move(E);
}

void SetTheory::addFieldExpander(StringRef ClassName, StringRef FieldName) {
  addExpander(ClassName, std::make_unique<FieldExpander>(FieldName));
}

void SetTheory::evaluate(const Init *Expr, RecSet &Elts, ArrayRef<SMLoc> Loc) {
  // A def in a list can be a just an element, or it may expand.
  if (const auto *Def = dyn_cast<DefInit>(Expr)) {
    if (const RecVec *Result = expand(Def->getDef()))
      return Elts.insert_range(*Result);
    Elts.insert(Def->getDef());
    return;
  }

  // Lists simply expand.
  if (const auto *LI = dyn_cast<ListInit>(Expr))
    return evaluate(LI->begin(), LI->end(), Elts, Loc);

  // Anything else must be a DAG.
  const auto *DagExpr = dyn_cast<DagInit>(Expr);
  if (!DagExpr)
    PrintFatalError(Loc, "Invalid set element: " + Expr->getAsString());
  const auto *OpInit = dyn_cast<DefInit>(DagExpr->getOperator());
  if (!OpInit)
    PrintFatalError(Loc, "Bad set expression: " + Expr->getAsString());
  auto I = Operators.find(OpInit->getDef()->getName());
  if (I == Operators.end())
    PrintFatalError(Loc, "Unknown set operator: " + Expr->getAsString());
  I->second->apply(*this, DagExpr, Elts, Loc);
}

const RecVec *SetTheory::expand(const Record *Set) {
  // Check existing entries for Set and return early.
  ExpandMap::iterator I = Expansions.find(Set);
  if (I != Expansions.end())
    return &I->second;

  // This is the first time we see Set. Find a suitable expander.
  for (const Record *SuperClass : Set->getSuperClasses()) {
    // Skip unnamed superclasses.
    if (!isa<StringInit>(SuperClass->getNameInit()))
      continue;
    auto I = Expanders.find(SuperClass->getName());
    if (I == Expanders.end())
      continue;
    // This breaks recursive definitions.
    RecVec &EltVec = Expansions[Set];
    RecSet Elts;
    I->second->expand(*this, Set, Elts);
    EltVec.assign(Elts.begin(), Elts.end());
    return &EltVec;
  }

  // Set is not expandable.
  return nullptr;
}
