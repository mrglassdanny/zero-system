
#include <vector>

#include "../core/tensor.cuh"

namespace zero_v2
{
    using namespace core;

    namespace cluster
    {
        class KMeans
        {
        private:
            int cluster_cnt;
            int feature_cnt;
            Tensor *clusters;

        public:
            KMeans(int cluster_cnt, int feature_cnt);
            KMeans(const KMeans &src);
            KMeans(const char *path);
            ~KMeans();

            void print();

            void save(const char *path);

            void initialize_clusters(Tensor *x);
            void reset_clusters();

            float train(Tensor *x);
            Tensor *predict(Tensor *x);

            static void save_best(Tensor *x, int cluster_cnt, int iter_cnt, const char *path);
        };
    }
}