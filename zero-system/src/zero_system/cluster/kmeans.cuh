
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

            void set_clusters(Tensor *x);
            float assign_inputs_to_clusters(Tensor *x, Tensor *assignments);
            void update_clusters(Tensor *x, Tensor *assignments);
            void reset_clusters();

        public:
            KMeans(int cluster_cnt, int feature_cnt);
            KMeans(const KMeans &src);
            KMeans(const char *path);
            ~KMeans();

            void print();

            void dump(const char *path);

            float train(Tensor *x, int train_chk_freq);
            Tensor *get_input_assignments(Tensor *x);
        };
    }
}