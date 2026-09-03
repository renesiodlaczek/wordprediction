# Shiny prototype: live next-word suggestions from the tuned n-gram model.
#
# Run with shiny::runApp(".") or the "Run App" button in Positron/RStudio.
# Expects to run with the working directory at the project root (so
# "R/..." and "data/processed/..." resolve) -- true by default when this
# file itself is launched as the app.
#
# Uses the interpolation scorer selected in R/04_ngram_model.R (see
# data/processed/model_config.rds for the tuned lambdas/pruning settings),
# reapplied here directly against the already-pruned model_*.rds lookup
# tables. That's a lighter-weight approximation of the scoring used for
# evaluation there (which draws on unpruned tables for exact next-word
# probabilities) -- an appropriate tradeoff for an interactive prototype
# where we only need to rank a handful of retrieved candidates, not score
# the whole vocabulary.

library(shiny)
library(bslib)
library(data.table)
library(stringr)

source("R/02_tokenize_clean.R")  # provides clean_lines()

model_dir <- "data/processed"
unigram_tab    <- readRDS(file.path(model_dir, "model_unigram.rds"))
bigram_tab     <- readRDS(file.path(model_dir, "model_bigram.rds"))
trigram_tab    <- readRDS(file.path(model_dir, "model_trigram.rds"))
fourgram_tab   <- readRDS(file.path(model_dir, "model_fourgram.rds"))
total_unigrams <- readRDS(file.path(model_dir, "model_total_unigrams.rds"))
model_config   <- readRDS(file.path(model_dir, "model_config.rds"))

setDT(unigram_tab); setkey(unigram_tab, word)
unigram_top <- copy(unigram_tab)[order(-N)]
setDT(bigram_tab); setkey(bigram_tab, prefix)
setDT(trigram_tab); setkey(trigram_tab, prefix)
setDT(fourgram_tab); setkey(fourgram_tab, prefix)

tables_by_order <- list(bigram_tab, trigram_tab, fourgram_tab)  # context length 1, 2, 3
lambdas <- model_config$lambdas  # weights for unigram/bigram/trigram/4-gram, tuned on the dev set

`%||%` <- function(a, b) if (length(a) == 0 || is.na(a)) b else a

# Interpolated score for one candidate word given the preceding words.
#
# NB: the candidate-word argument is named `target_word`, not `word` --
# data.table's `[` resolves bare symbols against the table's own columns
# before the enclosing environment, and these tables have a `word` column,
# so an argument named `word` would silently (and incorrectly) resolve to
# that column instead.
interpolated_score <- function(words, target_word) {
  max_context <- min(3, length(words))
  score <- lambdas[1] * (unigram_tab[.(target_word)]$N %||% 0) / total_unigrams
  for (k in seq_len(max_context)) {
    context <- paste(tail(words, k), collapse = " ")
    row <- tables_by_order[[k]][.(context)]
    row <- row[word == target_word]
    if (nrow(row) == 1) score <- score + lambdas[k + 1] * row$count / row$prefix_total
  }
  score
}

#' Predict the next word(s) given the phrase typed so far.
predict_next_word <- function(phrase, top_n = 3) {
  words <- str_split(clean_lines(phrase), " ")[[1]]
  words <- words[words != ""]
  if (length(words) == 0) return(head(unigram_top, top_n)$word)

  max_context <- min(3, length(words))
  # Candidate pool: union of each applicable order's top candidates for this
  # context -- keeps scoring cheap by never touching the full vocabulary.
  candidates <- character(0)
  for (k in seq_len(max_context)) {
    context <- paste(tail(words, k), collapse = " ")
    matches <- tables_by_order[[k]][.(context)]
    if (!is.na(matches$word[1])) candidates <- c(candidates, matches$word)
  }
  candidates <- unique(candidates)
  if (length(candidates) == 0) return(head(unigram_top, top_n)$word)

  scores <- vapply(candidates, function(w) interpolated_score(words, w), numeric(1))
  candidates[order(-scores)][seq_len(min(top_n, length(candidates)))]
}

ui <- page_fillable(
  title = "Next-Word Predictor",
  theme = bs_theme(version = 5),
  card(
    card_header("Next-word predictor \u2014 capstone prototype"),
    p("Type a phrase below. Suggested next words update as you type; click one to add it to the phrase."),
    textAreaInput("phrase", label = NULL, rows = 3, width = "100%",
                   placeholder = "e.g. thanks for the"),
    uiOutput("suggestion_buttons"),
    tags$p(
      class = "text-muted small mt-3",
      "Model: n-gram linear interpolation (unigram + bigram + trigram + four-gram), tuned on a 1,000,000-line sample of English Twitter/blogs/news text. See report/milestone_report.qmd for methodology and evaluation."
    )
  )
)

server <- function(input, output, session) {
  suggestions <- reactive({
    predict_next_word(input$phrase, top_n = 3)
  })

  output$suggestion_buttons <- renderUI({
    words <- suggestions()
    if (length(words) == 0) return(NULL)
    div(
      class = "d-flex flex-wrap gap-2 mt-2",
      lapply(seq_along(words), function(i) {
        actionButton(paste0("sugg_", i), words[i], class = "btn-outline-primary")
      })
    )
  })

  append_word <- function(i) {
    words <- isolate(suggestions())
    if (length(words) >= i) {
      current <- input$phrase %||% ""
      sep <- if (nzchar(trimws(current))) " " else ""
      updateTextAreaInput(session, "phrase", value = paste0(current, sep, words[i], " "))
    }
  }
  observeEvent(input$sugg_1, append_word(1))
  observeEvent(input$sugg_2, append_word(2))
  observeEvent(input$sugg_3, append_word(3))
}

shinyApp(ui, server)
