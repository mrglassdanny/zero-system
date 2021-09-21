
#include <vector>

#include "../core/tensor.cuh"

namespace zero
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

            void dump(const char *path);

            void set_clusters(Tensor *x);
            void reset_clusters();

            float train(Tensor *x);
            Tensor *predict(Tensor *x);

            static void dump_best(Tensor *x, int cluster_cnt, int iter_cnt, const char *path);
        };
    }
}