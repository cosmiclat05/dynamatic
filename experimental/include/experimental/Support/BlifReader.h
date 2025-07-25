//===- BlifReader.h - Exp. support for MAPBUF -----------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the functionality for reading a BLIF file into data
// structures.
//
//===-----------------------------------------------------------------===//

#ifndef EXPERIMENTAL_SUPPORT_BLIF_READER_H
#define EXPERIMENTAL_SUPPORT_BLIF_READER_H

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
#include "gurobi_c++.h"

#include "dynamatic/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Value.h"
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

using namespace mlir;

namespace dynamatic {
namespace experimental {

class LogicNetwork;

struct MILPVarsSubjectGraph {
  GRBVar tIn;
  GRBVar tOut;
  GRBVar bufferVar;
};

/// Represents a node in an And-Inverter Graph (AIG) circuit representation.
///
/// Each node can represent different circuit elements including:
/// - Primary inputs and outputs
/// - Latch (register) inputs and outputs
/// - Constant values (0 or 1)
/// - Blackbox output nodes
/// - Channel edge nodes
///
/// The node maintains its connectivity through fanin and fanout relationships
/// with other nodes, stored as pointer sets.
/// Node identity and ordering is determined by the unique name attribute.
///
/// The class integrates with MILP through the gurobiVars member for buffer
/// placement.
class Node {
public:
  bool isBlackboxOutput = false;
  bool isChannelEdge = false;
  bool isConstZero = false;
  bool isConstOne = false;
  bool isInput = false;
  bool isOutput = false;
  bool isLatchInput = false;
  bool isLatchOutput = false;
  Value nodeMLIRValue; // MLIR Value associated with the node, if any

  MILPVarsSubjectGraph *gurobiVars;
  std::set<Node *> fanins = {};
  std::set<Node *> fanouts = {};
  std::string function;
  std::string name;

  Node() = default;
  Node(const std::string &name, LogicNetwork *parent)
      : gurobiVars(new MILPVarsSubjectGraph()), name(name) {}

  // Adds a fanin node to the current node.
  void addFanin(Node *node) { fanins.insert(node); }
  // Adds a fanout node to the current node.
  void addFanout(Node *node) { fanouts.insert(node); }

  // Adds an edge between two nodes, updating fanin and fanout relationships.
  static void addEdge(Node *fanin, Node *fanout) {
    fanout->addFanin(fanin);
    fanin->addFanout(fanout);
  }

  // Configures the node as a latch based on the input and output nodes.
  static void configureLatch(Node *regInputNode, Node *regOutputNode) {
    regInputNode->isLatchInput = true;
    regOutputNode->isLatchOutput = true;
  }

  // Replaces an existing fanin with a new one.
  void replaceFanin(Node *oldFanin, Node *newFanin) {
    fanins.erase(oldFanin);
    fanins.insert(newFanin);
  }

  // Connects two nodes by setting the pointer of current node to the previous
  // node. This function is used to merge different LogicNetwork objects. Input
  // node of one LogicNetwork object is connected to the output node of
  // LogicNetwork object that comes before it.
  static void connectNodes(Node *currentNode, Node *previousNode, Value channel) {
    currentNode->nodeMLIRValue = channel;
    previousNode->nodeMLIRValue = channel;

    // Once Input/Output Nodes are connected, they should not be Input/Output in
    // the BLIF, but just become internal Nodes
    currentNode->convertIOToChannel();
    previousNode->convertIOToChannel();

    if (previousNode->isBlackboxOutput) {
      previousNode->isInput = true;
    }

    for (auto &fanout : currentNode->fanouts) {
      previousNode->addFanout(fanout);
      fanout->replaceFanin(currentNode, previousNode);
    }

    // Reverse the naming for ready signals
    if (previousNode->name.find("ready") != std::string::npos) {
      previousNode->name = currentNode->name;
    }
  }

  // Configures the node based on the type of I/O node.
  void configureIONode(const std::string &type);

  // Configures the node as a constant node based on the function.
  void configureConstantNode();

  bool isPrimaryInput() const {
    return (isConstOne || isConstZero || isInput || isLatchOutput ||
            isBlackboxOutput);
  }
  bool isPrimaryOutput() const { return (isOutput || isLatchInput); }

  // Used to merge I/O nodes. I/O is set false and isChannelEdge is set to true
  // so that the node can be considered as a dataflow graph edge.
  void convertIOToChannel();

  std::string str() const { return name; }

  ~Node() {
    delete gurobiVars;
    fanins.clear();
    fanouts.clear();
  }

  friend class LogicNetwork;
};

// Hash and Equality functions for Node pointers, used in unordered maps
struct NodePtrHash {
  std::size_t operator()(const Node *node) const {
    return std::hash<std::string>()(node->name);
  }
};

struct NodePtrEqual {
  bool operator()(const Node *lhs, const Node *rhs) const {
    return lhs->name == rhs->name;
  }
};

struct NodePairHash {
    std::size_t operator()(const std::pair<Node*, Node*>& p) const {
        auto h1 = NodePtrHash{}(p.first);
        auto h2 = NodePtrHash{}(p.second);
        
        // Simple hash function using bit shifting and XOR
        return h1 ^ (h2 << 1);
    }
};

/// Manages a collection of interconnected nodes representing a
/// circuit, maintaining relationships between nodes including latches and
/// topological ordering. The class provides functionality for:
/// - Node management (creation, modification, lookup)
/// - Circuit topology analysis (path finding, traversal)
/// - Circuit element categorization (inputs, outputs, channels)
/// - BLIF file generation
///
/// Node uniqueness is enforced through automatic name conflict resolution.
/// Memory ownership of nodes is maintained by this class through the nodes map.
class LogicNetwork {
private:
  std::vector<std::pair<Node *, Node *>> latches;
  std::unordered_map<std::string, Node *> nodes;
  std::vector<Node *> nodesTopologicalOrder;

public:
  LogicNetwork() = default;
  std::string moduleName;

  // Add input/output nodes to the circuit. Calls configureIONode on the Node
  // to set I/O type.
  void addIONode(const std::string &name, const std::string &type);

  // Add latch nodes to the circuit. Calls configureLatch on the Node to set
  // I/O type.
  void addLatch(const std::string &inputName, const std::string &outputName);

  // Add a constant node to the circuit. The function takes a vector of nodes
  // with only one element (ensured by parser) and a boolean function.
  void addConstantNode(const std::vector<std::string> &nodes,
                       const std::string &function);

  // Add a logic gate to the circuit. The function takes a vector of node
  // names, parsed from a file, and a boolean function. Sets the fanins and
  // fanouts, and calls Node functions to configure the fanout node.
  void addLogicGate(const std::vector<std::string> &nodes,
                    const std::string &function);

  // Adds a node to the circuit data structure, resolving name conflicts. If a
  // node with the same name already exists, counter value is appended to the
  // name. This function is used to add node that was already created to a
  // LogicNetwork object. Used to merge different LogicNetwork objects.
  Node *addNode(Node *node);

  // Creates a new Node and returns it. If a Node with the same name already
  // exists, the existing Node is returned. This function is used while parsing
  // a BLIF file.
  Node *createNode(const std::string &name);

  // Finds the path from "start" to "end" using bfs.
  std::vector<Node *> findPath(Node *start, Node *end);

  // Returns all of the Nodes.
  std::set<Node *> getAllNodes();

  // Returns the Nodes that correspond to Dataflow Graph Channels Edges.
  std::set<Node *> getChannels();

  // Returns latch vector.
  std::vector<std::pair<Node *, Node *>> getLatches() const { return latches; }

  // Returns the Primary Inputs by looping over all the Nodes and checking if
  // they are Primary Inputs by calling isPrimaryInput member function from
  // Node class.
  std::set<Node *> getPrimaryInputs();

  // Returns the Primary Outputs by looping over all the Nodes and checking
  // if they are Primary Outputs by calling isPrimaryOutput member function from
  // Node class.
  std::set<Node *> getPrimaryOutputs();

  // Returns the Nodes in topological order. Nodes were sorted in topological
  // order when LogicNetwork class is instantiated.
  std::vector<Node *> getNodesInTopologicalOrder() {
    return nodesTopologicalOrder;
  }

  // Returns Inputs of the Blif file.
  std::set<Node *> getInputs();

  // Returns Outputs of the Blif file.
  std::set<Node *> getOutputs();

  // Helper function for traverseNodes. Sorts the Nodes using dfs.
  void topologicalOrderUtil(Node *node, std::set<Node *> &visitedNodes,
                            std::vector<Node *> &order);

  // Traverses the nodes, sorting them into topological order from
  // Primary Inputs to Primary Outputs.
  void generateTopologicalOrder();
};

/// Generates a file in .blif format using the LogicNetwork data structure.
class BlifWriter {
public:
  BlifWriter() = default;
  void writeToFile(LogicNetwork &network, const std::string &filename);
};

/// Parses Berkeley Logic Interchange Format (BLIF) files into LogicNetwork
/// class.
class BlifParser {
public:
  BlifParser() = default;
  experimental::LogicNetwork *parseBlifFile(const std::string &filename);
};

} // namespace experimental
} // namespace dynamatic

#endif // DYNAMATIC_GUROBI_NOT_INSTALLED
#endif // EXPERIMENTAL_SUPPORT_BLIF_READER_H
