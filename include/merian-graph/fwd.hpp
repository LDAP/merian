#pragma once

#include <memory>

namespace merian {

// Forward declarations for node graph system
class Node;
class Connector;
class ConnectorInput;
class ConnectorOutput;
class Graph;
class GraphRun;
class Resource;
class NodeIO;
class NodeIOLayout;
class Graph;

// Handle types
using NodeHandle = std::shared_ptr<Node>;
using ConnectorHandle = std::shared_ptr<Connector>;
using InputConnectorHandle = std::shared_ptr<ConnectorInput>;
using OutputConnectorHandle = std::shared_ptr<ConnectorOutput>;
using GraphHandle = std::shared_ptr<Graph>;
using GraphRunHandle = std::shared_ptr<GraphRun>;
using GraphHandle = std::shared_ptr<Graph>;

} // namespace merian
