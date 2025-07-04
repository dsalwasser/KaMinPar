#include <iostream>
#include <vector>

#include <kaminpar-shm/kaminpar.h>

using namespace kaminpar;
using namespace kaminpar::shm;

std::string print_node_id(const NodeID id, const std::vector<BlockID> &partition) {
  const std::array<std::string, 4> colors = {
      "\033[0;31m", "\033[0;32m", "\033[0;33m", "\033[0;34m"
  };
  const std::string reset = "\033[0m";

  const std::string name = (id < 10) ? "0" + std::to_string(id) : std::to_string(id);

  if (partition.empty()) {
    return name;
  } else {
    return colors[partition[id]] + name + reset;
  }
}

void render_graph(const std::vector<BlockID> &partition) {
  std::cout << "   " << print_node_id(0, partition) << "        " << print_node_id(4, partition)
            << std::endl;
  std::cout << "  /  \\      /  \\" << std::endl;
  std::cout << print_node_id(3, partition) << "    " << print_node_id(1, partition) << "--"
            << print_node_id(7, partition) << "    " << print_node_id(5, partition) << std::endl;
  std::cout << "  \\  /      \\  /" << std::endl;
  std::cout << "   " << print_node_id(2, partition) << "        " << print_node_id(6, partition)
            << std::endl;
  std::cout << "   ||        ||" << std::endl;
  std::cout << "   " << print_node_id(12, partition) << "        " << print_node_id(8, partition)
            << std::endl;
  std::cout << "  /  \\      /  \\" << std::endl;
  std::cout << print_node_id(15, partition) << "    " << print_node_id(13, partition) << "--"
            << print_node_id(11, partition) << "    " << print_node_id(9, partition) << std::endl;
  std::cout << "  \\  /      \\  /" << std::endl;
  std::cout << "   " << print_node_id(14, partition) << "        " << print_node_id(10, partition)
            << std::endl;
}

int main() {
  //    00        04
  //   /  \      /  \
  // 03    01--07    05
  //   \  /      \  /
  //    02        06
  //    ||        ||
  //    12        08
  //   /  \      /  \
  // 15    13--11    09
  //   \  /      \  /
  //    14        10
  const NodeID n = 16;
  const std::vector<EdgeID> xadj{0, 2, 5, 8, 10, 12, 14, 17, 20, 23, 25, 27, 30, 33, 36, 38, 40};
  const std::vector<NodeID> adjncy{
      1, 3, 0,  2, 7,  1, 12, 3, 0,  2,  5, 7,  4,  6,  5,  7,  8,  1,  4,  6,
      6, 9, 11, 8, 10, 9, 11, 8, 10, 13, 2, 13, 15, 11, 12, 14, 13, 15, 12, 14,
  };

  KaMinPar shm(4, create_default_context());
  shm.set_output_level(OutputLevel::QUIET);
  shm.copy_graph(xadj, adjncy);

  // Balanced bipartition with eps = 0.03 = 3%
  {
    shm.set_k(2);
    shm.set_uniform_max_block_weights(0.03);

    std::vector<shm::BlockID> partition(n);
    const EdgeWeight cut = shm.compute_partition(partition);

    std::cout << "Balanced 2-way partition: " << cut << " edges cut" << std::endl;
    render_graph(partition);
    std::cout << std::endl;
  }

  // Balanced 4-way partition with eps = 0.03 = 3%
  {
    shm.set_k(4);
    shm.set_uniform_max_block_weights(0.03);

    std::vector<shm::BlockID> partition(n);
    const EdgeWeight cut = shm.compute_partition(partition);

    std::cout << "Balanced 4-way partition: " << cut << " edges cut" << std::endl;
    render_graph(partition);
    std::cout << std::endl;
  }

  // 3-way partition where block 0 has <= 50.1% of the total weight, blocks 1 and 2 have <= 25.1% of
  // the total weight
  {
    shm.set_k(3);
    shm.set_relative_max_block_weights(std::vector<double>{0.51, 0.251, 0.251});

    std::vector<shm::BlockID> partition(n);
    const EdgeWeight cut = shm.compute_partition(partition);

    std::cout << "Relative max block weights {0.5, 0.25, 0.25} + 0.01: " << cut << " edges cut"
              << std::endl;
    render_graph(partition);
    std::cout << std::endl;
  }

  // 4-way partition where block 0 has at most two, block 1 and 2 at most three, and block 3 at most
  // ten nodes
  {
    shm.set_k(4);
    shm.set_absolute_max_block_weights(std::vector<BlockWeight>{2, 3, 3, 10});

    std::vector<shm::BlockID> partition(n);
    const EdgeWeight cut = shm.compute_partition(partition);

    std::cout << "Absolute max block weights {2, 3, 3, 10}: " << cut << " edges cut" << std::endl;
    render_graph(partition);
    std::cout << std::endl;
  }
}
