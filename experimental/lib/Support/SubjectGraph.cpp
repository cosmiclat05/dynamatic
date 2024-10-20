#include "experimental/Support/SubjectGraph.h"

using namespace dynamatic::experimental;

void BaseSubjectGraph::assignSignals(ChannelSignals &signals, Node *node,
                   const std::string &nodeName) {
  if (nodeName.find("valid") != std::string::npos) {
    signals.validSignal = node;
  } else if (nodeName.find("ready") != std::string::npos) {
    signals.readySignal = node;
  } else {
    signals.dataSignals.push_back(node);
  }
};