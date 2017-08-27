#include <set>
#include <map>
#include <vector>


template <typename ClassId, typename SampleId>
class RankingItem {
  static_assert(std::is_scalar<SampleId>::value, "`SampleId` is not a scalar type");

  public: using class_id_t = ClassId;
  public: using sample_id_t = SampleId;

  public: RankingItem(const ClassId c_id, const SampleId s_id, const double score):
      classId(c_id), sampleId(s_id), score(score) {
  }

  public: ClassId getClass() const noexcept {
    return this->classId;
  }

  public: SampleId getSampleId() const noexcept {
    return this->sampleId;
  }

  public: double getScore() const noexcept {
    return this->score;
  }

  public: void setScore(double value) noexcept {
    this->score = value;
  }

  private: ClassId classId;
  private: SampleId sampleId;
  private: double score;
};

template <typename RankingItem_t>
class SimilarityRanking {
  // TODO: check if it is correct type
  // static_assert(std::is_base_of<::RankingItem, RankingItem_t>::value, "`RankingItem_t` is not an instance of RankingItem<Cid, Sid>");

  public: using ClassId = typename RankingItem_t::class_id_t;
  public: using SampleId = typename RankingItem_t::sample_id_t;

  public: using Container_t = std::map<ClassId, std::vector<RankingItem_t>>;

  public: size_t size() noexcept {
    std::lock_guard<std::mutex> lock(this->itemsLock);
    size_t count = 0;

    for (auto &class_i : this->items)
      count += class_i.second.size();

    return count;
  }

  public: void addElement(const RankingItem_t &ri) {
    std::lock_guard<std::mutex> lock(this->itemsLock);

    auto it = this->items.find(ri.getClass());
    if (it != this->items.end())
      it->second.push_back(ri);
    else {
      this->items.emplace(ri.getClass(), std::vector<RankingItem_t>{ri});
    }
  }

  public: std::vector<RankingItem_t> getTop(const size_t n = 1) {
    std::lock_guard<std::mutex> lock(this->itemsLock);

    // put all classes in a big heap and get a top (not optimal but will work for now)
    std::set<RankingItem_t, LessScoreCmp> heap;

    for (auto &class_i : this->items)
      for (auto &ri : class_i.second)
        heap.insert(ri);

    auto num = std::min(n, heap.size());
    std::vector<RankingItem_t> result(heap.rbegin(), std::next(heap.rbegin(), num));

    return result;
  }

  public: void supressNonMaximum(const SampleId radius) {
    std::lock_guard<std::mutex> lock(this->itemsLock);

    for (auto &class_i : this->items) {
      auto &rankItems = class_i.second;

      auto isNotLocalMaximum = [&radius, &rankItems] (const RankingItem_t &item) {
        bool isMaximum = true;
        auto offset = &item - &*rankItems.cbegin();
        auto it = std::next(rankItems.begin(), offset);

        auto begin = std::max(rankItems.begin(), std::prev(it, radius));
        auto end = std::min(rankItems.end(), std::next(it, radius+1));
        for (auto jt = begin; jt != end; jt = std::next(jt))
          isMaximum &= (jt->getScore() <= it->getScore());

        return !isMaximum;
      };

      rankItems.erase(std::remove_if(rankItems.begin(), rankItems.end(), isNotLocalMaximum), rankItems.end());
      // non-maximum supression should't remove all items from soem class, so no empty classes cleanup needed
    }
  }

  public: double getMaxScore() {
    double max_score = std::accumulate(this->begin(), this->end(), std::numeric_limits<double>::lowest(),
        [](const double acc, const RankingItem_t &ri) {
          return std::max(ri.getScore(), acc);
        });

    return max_score;
  }

  // in l_infinity sense
  public: double normalize() {
    std::lock_guard<std::mutex> lock(this->itemsLock);

    auto max_score = this->getMaxScore();

    for (auto &class_i : this->items)
      for (auto &ri : class_i.second)
        ri.setScore(ri.getScore() / max_score);

    return max_score;
  }

  public: void filter(const double minLevel) {
    std::lock_guard<std::mutex> lock(this->itemsLock);

    for (auto &class_i : this->items) {
      auto &rankItems = class_i.second;
      
      auto isItemBad = [&minLevel] (const RankingItem_t &item) {
        return (item.getScore() < minLevel);
      };

      rankItems.erase(std::remove_if(rankItems.begin(), rankItems.end(), isItemBad), rankItems.end());
    }

    this->removeEmptyClasses();
  }

  public: std::vector<double> getHistogram() {
    std::vector<double> result;
    for (auto sample_it = this->begin() ; sample_it != this->end(); ++sample_it) {
      result.push_back(sample_it->getScore());
    }

    return result;
  }

  public: class Iterator : public std::iterator<std::forward_iterator_tag, RankingItem_t> {
    public: typename Container_t::iterator class_it;
    public: typename Container_t::iterator class_it_end;
    public: typename Container_t::mapped_type::iterator sample_it;

    public: Iterator() = default;

    public: Iterator(decltype(class_it) c_it, decltype(class_it) c_it_end, decltype(sample_it) s_it):
        class_it(c_it), class_it_end(c_it_end), sample_it(s_it) {
    }

    public: Iterator &operator ++ () {
      if (this->sample_it != std::prev(this->class_it->second.end()))
        ++(this->sample_it);
      else {
        if (this->class_it != std::prev(this->class_it_end))
          this->sample_it = (++this->class_it)->second.begin();
        else
          ++(this->sample_it); // go for end()
      }

      return *this;
    }

    public: Iterator operator ++ (int) {
      Iterator retval = *this;
      this->operator ++ ();

      return retval;
    }

    public: bool operator == (Iterator it) const {
      return (this->class_it == it.class_it && this->sample_it == it.sample_it);
    }

    public: bool operator != (Iterator it) const {
      return !(*this == it);
    }

    RankingItem_t &operator * () const {
      return *sample_it;
    }

    RankingItem_t *operator -> () const {
      return &*sample_it;
    }
  };

  public: Iterator begin() {
    assert(!this->items.empty());

    return Iterator{this->items.begin(), this->items.end(), this->items.begin()->second.begin()};
  }

  public: Iterator end() {
    assert(!this->items.empty());

    Iterator it;
    it.class_it = std::prev(this->items.end());
    it.sample_it = it.class_it->second.end();

    return it;
  }

  protected: void removeEmptyClasses() {
    for(auto it = this->items.begin(); it != this->items.end();) {
      if (it->second.size() == 0)
        it = this->items.erase(it);
      else
        it = std::next(it);
    }
  }

  private: class LessScoreCmp {
    public: bool operator()(const RankingItem_t &a, const RankingItem_t &b) const {
      return (a.getScore() < b.getScore());
    }
  };

  private: std::mutex itemsLock;
  // TODO: use set instead of vector;
  private: Container_t items;
};
