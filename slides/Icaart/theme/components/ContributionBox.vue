<template>
  <div class="contribution-box">
    <div class="contribution-header">Contributions</div>
    <div class="contribution-body">
      <slot />
      <div v-if="$slots.footnote" class="contribution-footnote">
        <hr />
        <slot name="footnote" />
      </div>
    </div>
  </div>
</template>

<style scoped>
.contribution-box {
  border: 2px solid var(--slidev-theme-primary, #003678);
  border-radius: 6px;
  overflow: hidden;
  margin: 0.5em 0;
}

.contribution-header {
  background: var(--slidev-theme-primary, #003678);
  color: white;
  font-weight: 700;
  font-size: 1.05em;
  padding: 0.3em 0.75em;
}

.contribution-body {
  padding: 0.6em 0.75em;
}

.contribution-body :deep(p) {
  margin: 0.35em 0;
  line-height: 1.5;
}

.contribution-body :deep(.c-label) {
  font-weight: 700;
  color: var(--slidev-theme-primary, #003678);
  margin-right: 0.25em;
}

.contribution-footnote {
  margin-top: 0.4em;
  font-size: 0.8em;
  opacity: 0.75;
}

.contribution-footnote hr {
  border: none;
  border-top: 1px solid currentColor;
  opacity: 0.3;
  margin: 0.4em 0;
  width: 40%;
}
</style>
