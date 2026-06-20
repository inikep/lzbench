// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "tools/io/InputSetFileOrDir.h"

#include <filesystem>

#include "tools/io/InputFile.h"

namespace openzl::tools::io {

class InputSetFileOrDir::IteratorStateSingleFile
        : public InputSet::IteratorState {
   public:
    explicit IteratorStateSingleFile(const InputSetFileOrDir& isfd)
            : isfd_(isfd), input_(std::make_shared<InputFile>(isfd_.path_))
    {
    }

    std::unique_ptr<IteratorState> copy() const override
    {
        return std::make_unique<IteratorStateSingleFile>(*this);
    }

    IteratorState& operator++() override
    {
        if (input_) {
            input_.reset();
        }
        return *this;
    }

    const std::shared_ptr<Input>& operator*() const override
    {
        return input_;
    }

    bool operator==(const IteratorState& o) const override
    {
        auto ptr = dynamic_cast<const IteratorStateSingleFile*>(&o);
        if (ptr == nullptr) {
            return false;
        }
        return (&isfd_ == &ptr->isfd_) && (!!input_ == !!ptr->input_);
    }

   private:
    const InputSetFileOrDir& isfd_;
    std::shared_ptr<Input> input_;
};

InputSetFileOrDir::InputSetFileOrDir(std::string path, bool recursive)
        : InputSetDir(std::move(path), recursive),
          is_dir_(std::filesystem::status(path_).type()
                  == std::filesystem::file_type::directory)
{
}

std::unique_ptr<InputSet::IteratorState> InputSetFileOrDir::begin_state() const
{
    if (is_dir_) {
        return InputSetDir::begin_state();
    } else {
        return std::make_unique<IteratorStateSingleFile>(*this);
    }
}

} // namespace openzl::tools::io
