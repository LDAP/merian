// Parsing logic derived from pbrt-v4 (https://github.com/mmp/pbrt-v4),
// Copyright (c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys. Apache-2.0.
#pragma once

#include <fmt/format.h>

#include <cassert>
#include <cctype>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace merian::pbrt {

struct PBRTParseError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

class Tokenizer {
  public:
    struct Token {
        enum class Kind : uint8_t { End, String, LBracket, RBracket, Bare };
        Kind kind = Kind::End;
        // String tokens carry the content without quotes.
        std::string_view text;
    };

    Tokenizer(std::string source, std::filesystem::path path) {
        frames.emplace_back(std::move(source), std::move(path));
    }

    const Token& peek() {
        if (!peeked) {
            peeked = scan();
        }
        return *peeked;
    }

    Token next() {
        if (peeked) {
            const Token token = *peeked;
            peeked.reset();
            return token;
        }
        return scan();
    }

    // Callers must have consumed the include filename first (no buffered token).
    void push_include(std::string source, std::filesystem::path path) {
        assert(!peeked);
        frames.emplace_back(std::move(source), std::move(path));
    }

    std::string location() const {
        if (frames.empty()) {
            return "<end of file>";
        }
        const Frame& frame = frames.back();
        return fmt::format("{}:{}", frame.path.string(), frame.line);
    }

  private:
    struct Frame {
        // Heap-stable so returned token views survive frame moves.
        std::unique_ptr<std::string> buffer;
        std::filesystem::path path;
        size_t pos = 0;
        int line = 1;

        Frame(std::string buffer, std::filesystem::path path)
            : buffer(std::make_unique<std::string>(std::move(buffer))), path(std::move(path)) {}
    };

    Token scan() {
        while (!frames.empty()) {
            Frame& frame = frames.back();
            const std::string& buf = *frame.buffer;

            while (frame.pos < buf.size()) {
                const char c = buf[frame.pos];
                if (c == '\n') {
                    frame.line++;
                    frame.pos++;
                } else if (std::isspace(static_cast<unsigned char>(c)) != 0) {
                    frame.pos++;
                } else if (c == '#') {
                    while (frame.pos < buf.size() && buf[frame.pos] != '\n') {
                        frame.pos++;
                    }
                } else {
                    break;
                }
            }
            if (frame.pos >= buf.size()) {
                // Keep the buffer alive: previously returned tokens view into it.
                retired.emplace_back(std::move(frames.back()));
                frames.pop_back();
                continue;
            }

            const char c = buf[frame.pos];
            if (c == '[') {
                frame.pos++;
                return {Token::Kind::LBracket, "["};
            }
            if (c == ']') {
                frame.pos++;
                return {Token::Kind::RBracket, "]"};
            }
            if (c == '"') {
                const size_t start = ++frame.pos;
                while (frame.pos < buf.size() && buf[frame.pos] != '"' && buf[frame.pos] != '\n') {
                    frame.pos++;
                }
                if (frame.pos >= buf.size() || buf[frame.pos] != '"') {
                    throw PBRTParseError(fmt::format("unterminated string at {}", location()));
                }
                const Token token{Token::Kind::String,
                                  std::string_view(buf).substr(start, frame.pos - start)};
                frame.pos++;
                return token;
            }

            const size_t start = frame.pos;
            while (frame.pos < buf.size()) {
                const char b = buf[frame.pos];
                if (std::isspace(static_cast<unsigned char>(b)) != 0 || b == '[' || b == ']' ||
                    b == '"' || b == '#') {
                    break;
                }
                frame.pos++;
            }
            return {Token::Kind::Bare, std::string_view(buf).substr(start, frame.pos - start)};
        }
        return {};
    }

    std::vector<Frame> frames;
    std::vector<Frame> retired;
    std::optional<Token> peeked;
};

} // namespace merian::pbrt
