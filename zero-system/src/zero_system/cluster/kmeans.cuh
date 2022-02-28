
#include <vector>

#include "../core/mod.cuh"

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
            KMeans();
            KMeans(int cluster_cnt, int feature_cnt);
            KMeans(const KMeans &src);
            ~KMeans();

            void print();

            void load(const char *path);
            void save(const char *path);

            void initialize_clusters(Tensor *xs);
            void reset_clusters();

            float train(Tensor *xs);
            Tensor *predict(Tensor *xs);

            static float save_best(Tensor *xs, int cluster_cnt, int iter_cnt, const char *path);
            static void run_elbow_analysis(Tensor *xs, int cluster_cnt_lower, int cluster_cnt_upper,
                                           int iter_cnt, const char *csv_path);
        };
    }
}